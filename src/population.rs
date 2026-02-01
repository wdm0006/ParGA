//! Population management for genetic algorithms.
//!
//! A population is a collection of individuals that evolve over generations.

use crate::genome::Genome;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Configuration for a population.
#[derive(Debug, Clone, derive_builder::Builder)]
#[builder(setter(into))]
pub struct PopulationConfig {
    /// Number of individuals in the population.
    #[builder(default = "100")]
    pub size: usize,

    /// Length of each genome.
    pub genome_length: usize,

    /// Lower bounds for genes.
    #[builder(default = "None")]
    pub lower_bounds: Option<Vec<f64>>,

    /// Upper bounds for genes.
    #[builder(default = "None")]
    pub upper_bounds: Option<Vec<f64>>,
}

impl PopulationConfig {
    /// Creates a new builder for `PopulationConfig`.
    pub fn builder() -> PopulationConfigBuilder {
        PopulationConfigBuilder::default()
    }

    /// Returns bounds with defaults if not specified.
    pub fn bounds(&self) -> (Vec<f64>, Vec<f64>) {
        let lower = self
            .lower_bounds
            .clone()
            .unwrap_or_else(|| vec![-10.0; self.genome_length]);
        let upper = self
            .upper_bounds
            .clone()
            .unwrap_or_else(|| vec![10.0; self.genome_length]);
        (lower, upper)
    }
}

/// An individual in the population.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Individual<G: Genome> {
    /// The genome (genetic material).
    pub genome: G,

    /// Fitness value (None if not yet evaluated).
    #[serde(default)]
    pub fitness: Option<f64>,
}

impl<G: Genome> Individual<G> {
    /// Creates a new individual with the given genome.
    pub fn new(genome: G) -> Self {
        Self {
            genome,
            fitness: None,
        }
    }

    /// Creates a new individual with genome and fitness.
    pub fn with_fitness(genome: G, fitness: f64) -> Self {
        Self {
            genome,
            fitness: Some(fitness),
        }
    }

    /// Returns the fitness, or negative infinity if not evaluated.
    pub fn fitness_or_default(&self) -> f64 {
        self.fitness.unwrap_or(f64::NEG_INFINITY)
    }

    /// Invalidates the cached fitness (e.g., after mutation).
    pub fn invalidate_fitness(&mut self) {
        self.fitness = None;
    }
}

impl<G: Genome + Default> Default for Individual<G> {
    fn default() -> Self {
        Self::new(G::default())
    }
}

impl<G: Genome> PartialEq for Individual<G>
where
    G: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.genome == other.genome && self.fitness == other.fitness
    }
}

/// A population of individuals.
#[derive(Debug, Clone)]
pub struct Population<G: Genome> {
    individuals: Vec<Individual<G>>,
}

impl<G: Genome> Population<G> {
    /// Creates an empty population.
    pub fn new() -> Self {
        Self {
            individuals: Vec::new(),
        }
    }

    /// Creates a population with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            individuals: Vec::with_capacity(capacity),
        }
    }

    /// Creates a population from existing individuals.
    pub fn from_individuals(individuals: Vec<Individual<G>>) -> Self {
        Self { individuals }
    }

    /// Creates a random population.
    pub fn random<R: Rng>(rng: &mut R, size: usize, lower: &[f64], upper: &[f64]) -> Self {
        let len = lower.len().max(upper.len());
        let individuals = (0..size)
            .map(|_| Individual::new(G::random(rng, len, lower, upper)))
            .collect();
        Self { individuals }
    }

    /// Creates a random population from configuration.
    pub fn from_config<R: Rng>(rng: &mut R, config: &PopulationConfig) -> Self {
        let (lower, upper) = config.bounds();
        Self::random(rng, config.size, &lower, &upper)
    }

    /// Returns the number of individuals.
    pub fn len(&self) -> usize {
        self.individuals.len()
    }

    /// Returns true if the population is empty.
    pub fn is_empty(&self) -> bool {
        self.individuals.is_empty()
    }

    /// Returns a reference to the individuals.
    pub fn individuals(&self) -> &[Individual<G>] {
        &self.individuals
    }

    /// Returns a mutable reference to the individuals.
    pub fn individuals_mut(&mut self) -> &mut [Individual<G>] {
        &mut self.individuals
    }

    /// Adds an individual to the population.
    pub fn push(&mut self, individual: Individual<G>) {
        self.individuals.push(individual);
    }

    /// Removes and returns the last individual.
    pub fn pop(&mut self) -> Option<Individual<G>> {
        self.individuals.pop()
    }

    /// Gets an individual by index.
    pub fn get(&self, index: usize) -> Option<&Individual<G>> {
        self.individuals.get(index)
    }

    /// Gets a mutable reference to an individual by index.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut Individual<G>> {
        self.individuals.get_mut(index)
    }

    /// Returns the best individual (highest fitness).
    pub fn best(&self) -> Option<&Individual<G>> {
        self.individuals
            .iter()
            .filter(|i| i.fitness.is_some())
            .max_by(|a, b| {
                a.fitness
                    .partial_cmp(&b.fitness)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Returns the worst individual (lowest fitness).
    pub fn worst(&self) -> Option<&Individual<G>> {
        self.individuals
            .iter()
            .filter(|i| i.fitness.is_some())
            .min_by(|a, b| {
                a.fitness
                    .partial_cmp(&b.fitness)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Returns the best N individuals.
    pub fn best_n(&self, n: usize) -> Vec<&Individual<G>> {
        let mut sorted: Vec<_> = self
            .individuals
            .iter()
            .filter(|i| i.fitness.is_some())
            .collect();
        sorted.sort_by(|a, b| {
            b.fitness
                .partial_cmp(&a.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.into_iter().take(n).collect()
    }

    /// Returns the average fitness of the population.
    pub fn average_fitness(&self) -> Option<f64> {
        let fitnesses: Vec<_> = self.individuals.iter().filter_map(|i| i.fitness).collect();

        if fitnesses.is_empty() {
            None
        } else {
            Some(fitnesses.iter().sum::<f64>() / fitnesses.len() as f64)
        }
    }

    /// Returns the fitness variance.
    pub fn fitness_variance(&self) -> Option<f64> {
        let avg = self.average_fitness()?;
        let fitnesses: Vec<_> = self.individuals.iter().filter_map(|i| i.fitness).collect();

        if fitnesses.is_empty() {
            None
        } else {
            Some(fitnesses.iter().map(|f| (f - avg).powi(2)).sum::<f64>() / fitnesses.len() as f64)
        }
    }

    /// Sorts the population by fitness (descending - best first).
    pub fn sort_by_fitness(&mut self) {
        self.individuals.sort_by(|a, b| {
            b.fitness
                .partial_cmp(&a.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Returns an iterator over the individuals.
    pub fn iter(&self) -> impl Iterator<Item = &Individual<G>> {
        self.individuals.iter()
    }

    /// Returns a mutable iterator over the individuals.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Individual<G>> {
        self.individuals.iter_mut()
    }

    /// Truncates the population to the given size, keeping the best.
    pub fn truncate(&mut self, size: usize) {
        self.sort_by_fitness();
        self.individuals.truncate(size);
    }

    /// Merges another population into this one.
    pub fn merge(&mut self, other: Population<G>) {
        self.individuals.extend(other.individuals);
    }

    /// Clears all individuals from the population.
    pub fn clear(&mut self) {
        self.individuals.clear();
    }

    /// Returns statistics about the population.
    pub fn statistics(&self) -> PopulationStatistics {
        let fitnesses: Vec<_> = self.individuals.iter().filter_map(|i| i.fitness).collect();

        if fitnesses.is_empty() {
            return PopulationStatistics::default();
        }

        let min = fitnesses.iter().copied().fold(f64::INFINITY, f64::min);
        let max = fitnesses.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mean = fitnesses.iter().sum::<f64>() / fitnesses.len() as f64;
        let variance =
            fitnesses.iter().map(|f| (f - mean).powi(2)).sum::<f64>() / fitnesses.len() as f64;
        let std_dev = variance.sqrt();

        PopulationStatistics {
            size: self.individuals.len(),
            evaluated: fitnesses.len(),
            min_fitness: min,
            max_fitness: max,
            mean_fitness: mean,
            std_dev,
        }
    }
}

impl<G: Genome> Default for Population<G> {
    fn default() -> Self {
        Self::new()
    }
}

impl<G: Genome> IntoIterator for Population<G> {
    type Item = Individual<G>;
    type IntoIter = std::vec::IntoIter<Individual<G>>;

    fn into_iter(self) -> Self::IntoIter {
        self.individuals.into_iter()
    }
}

impl<'a, G: Genome> IntoIterator for &'a Population<G> {
    type Item = &'a Individual<G>;
    type IntoIter = std::slice::Iter<'a, Individual<G>>;

    fn into_iter(self) -> Self::IntoIter {
        self.individuals.iter()
    }
}

impl<G: Genome> std::ops::Index<usize> for Population<G> {
    type Output = Individual<G>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.individuals[index]
    }
}

impl<G: Genome> std::ops::IndexMut<usize> for Population<G> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.individuals[index]
    }
}

/// Statistics about a population.
#[derive(Debug, Clone, Default)]
pub struct PopulationStatistics {
    /// Total population size.
    pub size: usize,
    /// Number of individuals with evaluated fitness.
    pub evaluated: usize,
    /// Minimum fitness value.
    pub min_fitness: f64,
    /// Maximum fitness value.
    pub max_fitness: f64,
    /// Mean fitness value.
    pub mean_fitness: f64,
    /// Standard deviation of fitness.
    pub std_dev: f64,
}

impl std::fmt::Display for PopulationStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Population(size={}, evaluated={}, fitness: min={:.4}, max={:.4}, mean={:.4}, std={:.4})",
            self.size, self.evaluated, self.min_fitness, self.max_fitness, self.mean_fitness, self.std_dev
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::RealGenome;

    #[test]
    fn test_individual_creation() {
        let genome = RealGenome::new(vec![1.0, 2.0, 3.0]);
        let ind = Individual::new(genome);
        assert!(ind.fitness.is_none());
        assert_eq!(ind.genome.len(), 3);
    }

    #[test]
    fn test_individual_with_fitness() {
        let genome = RealGenome::new(vec![1.0, 2.0, 3.0]);
        let ind = Individual::with_fitness(genome, 42.0);
        assert_eq!(ind.fitness, Some(42.0));
    }

    #[test]
    fn test_population_random() {
        let mut rng = crate::rng::create_rng(Some(42));
        let lower = vec![-5.0, -5.0, -5.0];
        let upper = vec![5.0, 5.0, 5.0];
        let pop: Population<RealGenome> = Population::random(&mut rng, 10, &lower, &upper);

        assert_eq!(pop.len(), 10);
        for ind in &pop {
            for (i, &gene) in ind.genome.genes().iter().enumerate() {
                assert!(gene >= lower[i] && gene <= upper[i]);
            }
        }
    }

    #[test]
    fn test_population_best() {
        let mut pop: Population<RealGenome> = Population::new();
        pop.push(Individual::with_fitness(RealGenome::new(vec![1.0]), 10.0));
        pop.push(Individual::with_fitness(RealGenome::new(vec![2.0]), 30.0));
        pop.push(Individual::with_fitness(RealGenome::new(vec![3.0]), 20.0));

        let best = pop.best().unwrap();
        assert_eq!(best.fitness, Some(30.0));
    }

    #[test]
    fn test_population_statistics() {
        let mut pop: Population<RealGenome> = Population::new();
        pop.push(Individual::with_fitness(RealGenome::new(vec![1.0]), 10.0));
        pop.push(Individual::with_fitness(RealGenome::new(vec![2.0]), 20.0));
        pop.push(Individual::with_fitness(RealGenome::new(vec![3.0]), 30.0));

        let stats = pop.statistics();
        assert_eq!(stats.size, 3);
        assert_eq!(stats.evaluated, 3);
        assert_eq!(stats.min_fitness, 10.0);
        assert_eq!(stats.max_fitness, 30.0);
        assert!((stats.mean_fitness - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_population_sort() {
        let mut pop: Population<RealGenome> = Population::new();
        pop.push(Individual::with_fitness(RealGenome::new(vec![1.0]), 10.0));
        pop.push(Individual::with_fitness(RealGenome::new(vec![2.0]), 30.0));
        pop.push(Individual::with_fitness(RealGenome::new(vec![3.0]), 20.0));

        pop.sort_by_fitness();

        assert_eq!(pop[0].fitness, Some(30.0));
        assert_eq!(pop[1].fitness, Some(20.0));
        assert_eq!(pop[2].fitness, Some(10.0));
    }
}
