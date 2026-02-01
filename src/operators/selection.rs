//! Selection operators for genetic algorithms.
//!
//! Selection determines which individuals are chosen as parents for the next generation.

use crate::genome::Genome;
use crate::population::{Individual, Population};
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Trait for selection operators.
pub trait Selection<G: Genome> {
    /// Selects individuals from the population for breeding.
    fn select<R: Rng>(
        &self,
        population: &Population<G>,
        count: usize,
        rng: &mut R,
    ) -> Vec<Individual<G>>;
}

/// Available selection methods.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SelectionOperator {
    /// Tournament selection with configurable tournament size.
    Tournament(usize),

    /// Roulette wheel (fitness proportionate) selection.
    RouletteWheel,

    /// Rank-based selection.
    Rank,

    /// Random selection (baseline).
    Random,

    /// Truncation selection (select top k%).
    Truncation(f64),

    /// Stochastic universal sampling.
    StochasticUniversal,
}

impl Default for SelectionOperator {
    fn default() -> Self {
        Self::Tournament(3)
    }
}

impl<G: Genome + Clone> Selection<G> for SelectionOperator {
    fn select<R: Rng>(
        &self,
        population: &Population<G>,
        count: usize,
        rng: &mut R,
    ) -> Vec<Individual<G>> {
        match self {
            Self::Tournament(size) => tournament_selection(population, count, *size, rng),
            Self::RouletteWheel => roulette_selection(population, count, rng),
            Self::Rank => rank_selection(population, count, rng),
            Self::Random => random_selection(population, count, rng),
            Self::Truncation(ratio) => truncation_selection(population, count, *ratio),
            Self::StochasticUniversal => sus_selection(population, count, rng),
        }
    }
}

/// Tournament selection: randomly pick k individuals, select the best.
fn tournament_selection<G: Genome + Clone, R: Rng>(
    population: &Population<G>,
    count: usize,
    tournament_size: usize,
    rng: &mut R,
) -> Vec<Individual<G>> {
    let individuals = population.individuals();
    if individuals.is_empty() {
        return Vec::new();
    }

    (0..count)
        .map(|_| {
            let tournament: Vec<_> = (0..tournament_size)
                .map(|_| &individuals[rng.gen_range(0..individuals.len())])
                .collect();

            tournament
                .into_iter()
                .max_by(|a, b| {
                    a.fitness
                        .partial_cmp(&b.fitness)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .cloned()
                .unwrap_or_else(|| individuals[0].clone())
        })
        .collect()
}

/// Roulette wheel selection: probability proportional to fitness.
fn roulette_selection<G: Genome + Clone, R: Rng>(
    population: &Population<G>,
    count: usize,
    rng: &mut R,
) -> Vec<Individual<G>> {
    let individuals = population.individuals();
    if individuals.is_empty() {
        return Vec::new();
    }

    // Shift fitness values to be positive
    let min_fitness = individuals
        .iter()
        .filter_map(|i| i.fitness)
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0);

    let offset = if min_fitness < 0.0 {
        -min_fitness + 1.0
    } else {
        0.0
    };

    let total_fitness: f64 = individuals
        .iter()
        .filter_map(|i| i.fitness)
        .map(|f| f + offset)
        .sum();

    if total_fitness <= 0.0 {
        return random_selection(population, count, rng);
    }

    (0..count)
        .map(|_| {
            let target = rng.gen::<f64>() * total_fitness;
            let mut cumulative = 0.0;

            for ind in individuals {
                cumulative += ind.fitness.unwrap_or(0.0) + offset;
                if cumulative >= target {
                    return ind.clone();
                }
            }

            individuals
                .last()
                .cloned()
                .unwrap_or_else(|| individuals[0].clone())
        })
        .collect()
}

/// Rank-based selection: probability proportional to rank.
fn rank_selection<G: Genome + Clone, R: Rng>(
    population: &Population<G>,
    count: usize,
    rng: &mut R,
) -> Vec<Individual<G>> {
    let individuals = population.individuals();
    if individuals.is_empty() {
        return Vec::new();
    }

    let n = individuals.len();
    // Ranks: worst gets 1, best gets n
    let total_rank: usize = n * (n + 1) / 2;

    (0..count)
        .map(|_| {
            let target = rng.gen_range(0..total_rank);
            let mut cumulative = 0;

            // Assume population is sorted (best first), so assign ranks in reverse
            for (i, ind) in individuals.iter().enumerate() {
                cumulative += n - i; // Best individual gets rank n
                if cumulative > target {
                    return ind.clone();
                }
            }

            individuals[0].clone()
        })
        .collect()
}

/// Random selection: purely random (baseline).
fn random_selection<G: Genome + Clone, R: Rng>(
    population: &Population<G>,
    count: usize,
    rng: &mut R,
) -> Vec<Individual<G>> {
    let individuals = population.individuals();
    if individuals.is_empty() {
        return Vec::new();
    }

    (0..count)
        .map(|_| individuals[rng.gen_range(0..individuals.len())].clone())
        .collect()
}

/// Truncation selection: select from top k%.
fn truncation_selection<G: Genome + Clone>(
    population: &Population<G>,
    count: usize,
    ratio: f64,
) -> Vec<Individual<G>> {
    let individuals = population.individuals();
    if individuals.is_empty() {
        return Vec::new();
    }

    let cutoff = ((individuals.len() as f64) * ratio.clamp(0.1, 1.0)).ceil() as usize;
    let top = &individuals[..cutoff.min(individuals.len())];

    (0..count).map(|i| top[i % top.len()].clone()).collect()
}

/// Stochastic Universal Sampling: evenly spaced pointers on fitness wheel.
fn sus_selection<G: Genome + Clone, R: Rng>(
    population: &Population<G>,
    count: usize,
    rng: &mut R,
) -> Vec<Individual<G>> {
    let individuals = population.individuals();
    if individuals.is_empty() {
        return Vec::new();
    }

    // Shift fitness values to be positive
    let min_fitness = individuals
        .iter()
        .filter_map(|i| i.fitness)
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0);

    let offset = if min_fitness < 0.0 {
        -min_fitness + 1.0
    } else {
        0.0
    };

    let total_fitness: f64 = individuals
        .iter()
        .filter_map(|i| i.fitness)
        .map(|f| f + offset)
        .sum();

    if total_fitness <= 0.0 {
        return random_selection(population, count, rng);
    }

    let pointer_distance = total_fitness / count as f64;
    let start = rng.gen::<f64>() * pointer_distance;
    let mut selected = Vec::with_capacity(count);

    let mut cumulative = 0.0;
    let mut individual_idx = 0;

    for i in 0..count {
        let pointer = start + (i as f64) * pointer_distance;

        while cumulative < pointer && individual_idx < individuals.len() {
            cumulative += individuals[individual_idx].fitness.unwrap_or(0.0) + offset;
            if cumulative < pointer {
                individual_idx += 1;
            }
        }

        if individual_idx < individuals.len() {
            selected.push(individuals[individual_idx].clone());
        } else {
            selected.push(individuals.last().cloned().unwrap());
        }
    }

    selected
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::RealGenome;

    fn create_test_population() -> Population<RealGenome> {
        let individuals = vec![
            Individual {
                genome: RealGenome::new(vec![1.0]),
                fitness: Some(10.0),
            },
            Individual {
                genome: RealGenome::new(vec![2.0]),
                fitness: Some(20.0),
            },
            Individual {
                genome: RealGenome::new(vec![3.0]),
                fitness: Some(30.0),
            },
        ];
        Population::from_individuals(individuals)
    }

    #[test]
    fn test_tournament_selection() {
        let pop = create_test_population();
        let mut rng = crate::rng::create_rng(Some(42));
        let selected = tournament_selection(&pop, 5, 2, &mut rng);
        assert_eq!(selected.len(), 5);
    }

    #[test]
    fn test_roulette_selection() {
        let pop = create_test_population();
        let mut rng = crate::rng::create_rng(Some(42));
        let selected = roulette_selection(&pop, 5, &mut rng);
        assert_eq!(selected.len(), 5);
    }

    #[test]
    fn test_rank_selection() {
        let pop = create_test_population();
        let mut rng = crate::rng::create_rng(Some(42));
        let selected = rank_selection(&pop, 5, &mut rng);
        assert_eq!(selected.len(), 5);
    }

    #[test]
    fn test_truncation_selection() {
        let pop = create_test_population();
        let selected = truncation_selection(&pop, 5, 0.5);
        assert_eq!(selected.len(), 5);
    }
}
