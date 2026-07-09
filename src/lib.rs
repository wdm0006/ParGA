//! # `ParGA` - Parallel Genetic Algorithm Library
//!
//! A high-performance genetic algorithm library written in Rust with Python bindings.
//!
//! ## Features
//!
//! - **Multiple genome types**: Binary, Real-valued, and Permutation
//! - **Island model**: Parallel evolution with configurable migration
//! - **Flexible operators**: Selection, crossover, and mutation strategies
//! - **Built-in benchmarks**: Standard optimization test functions
//! - **Python bindings**: Seamless integration via `PyO3`
//!
//! ## Example
//!
//! ```rust
//! use parga::prelude::*;
//!
//! // Create and run the genetic algorithm with built-in Sphere function
//! let config = GaConfig::builder()
//!     .population_size(100)
//!     .genome_length(10)
//!     .generations(100)
//!     .build()
//!     .unwrap();
//!
//! // Use the built-in Sphere benchmark (minimizes sum of squares)
//! let mut ga: GeneticAlgorithm<RealGenome, _> = GeneticAlgorithm::new(config, Sphere);
//! let result = ga.run();
//!
//! println!("Best fitness: {}", result.best_fitness);
//! ```

#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::cognitive_complexity)]

pub mod error;
pub mod fitness;
pub mod genome;
pub mod island;
pub mod operators;
pub mod population;
pub mod rng;

#[cfg(feature = "python")]
pub mod python;

use rand::Rng;

pub use error::{Error, Result};
pub use fitness::FitnessFunction;
pub use genome::{BinaryGenome, Genome, PermutationGenome, RealGenome};
pub use island::{IslandConfig, IslandModel, MigrationTopology};
pub use operators::{
    crossover::{Crossover, CrossoverOperator},
    mutation::{Mutation, MutationOperator},
    selection::{Selection, SelectionOperator},
};
pub use population::{Individual, Population, PopulationConfig};

/// Configuration for the genetic algorithm.
#[derive(Debug, Clone, derive_builder::Builder)]
pub struct GaConfig {
    /// Number of individuals in the population.
    #[builder(default = "100")]
    pub population_size: usize,

    /// Length of each genome.
    pub genome_length: usize,

    /// Number of generations to evolve.
    #[builder(default = "100")]
    pub generations: usize,

    /// Mutation rate (probability of mutation per gene).
    #[builder(default = "0.01")]
    pub mutation_rate: f64,

    /// Crossover rate (probability of crossover).
    #[builder(default = "0.8")]
    pub crossover_rate: f64,

    /// Number of elite individuals to preserve.
    #[builder(default = "2")]
    pub elitism: usize,

    /// Tournament size for tournament selection.
    #[builder(default = "3")]
    pub tournament_size: usize,

    /// Lower bounds for real-valued genomes.
    #[builder(default = "None", setter(strip_option))]
    pub lower_bounds: Option<Vec<f64>>,

    /// Upper bounds for real-valued genomes.
    #[builder(default = "None", setter(strip_option))]
    pub upper_bounds: Option<Vec<f64>>,

    /// Random seed for reproducibility.
    #[builder(default = "None", setter(strip_option))]
    pub seed: Option<u64>,

    /// Stop early if best fitness hasn't improved for this many generations.
    /// None means no early stopping (run all generations).
    #[builder(default = "None", setter(strip_option))]
    pub early_stopping: Option<usize>,

    /// Reinitialize non-elite population when fitness stagnates for this many
    /// generations. Keeps the top `elitism` individuals and randomizes the rest.
    /// None means no restart (default).
    #[builder(default = "None", setter(strip_option))]
    pub restart_on_stagnation: Option<usize>,

    /// Number of local search iterations to apply to the best individual each
    /// generation. Each iteration tries a small random perturbation and keeps
    /// it if fitness improves. 0 or None means no local search (default).
    #[builder(default = "None", setter(strip_option))]
    pub local_search_iters: Option<usize>,

    /// Final mutation rate for linear decay. If set, the mutation rate linearly
    /// interpolates from `mutation_rate` to `mutation_rate_end` over the generations.
    #[builder(default = "None", setter(strip_option))]
    pub mutation_rate_end: Option<f64>,

    /// Number of worst individuals to replace with random ones each generation.
    /// Provides continuous diversity injection. None means no immigrants (default).
    #[builder(default = "None", setter(strip_option))]
    pub random_immigrants: Option<usize>,
}

impl GaConfig {
    /// Creates a new builder for `GaConfig`.
    pub fn builder() -> GaConfigBuilder {
        GaConfigBuilder::default()
    }

    /// Returns the bounds as tuples, using defaults if not specified.
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

/// Result of a genetic algorithm run.
#[derive(Debug, Clone)]
pub struct GaResult<G: Genome> {
    /// The best individual found.
    pub best_individual: Individual<G>,
    /// Best fitness value.
    pub best_fitness: f64,
    /// Number of generations evolved.
    pub generations: usize,
    /// Fitness history (best fitness per generation).
    pub fitness_history: Vec<f64>,
    /// Whether the algorithm converged.
    pub converged: bool,
}

/// Main genetic algorithm executor.
pub struct GeneticAlgorithm<G, F>
where
    G: Genome,
    F: FitnessFunction<G>,
{
    config: GaConfig,
    population: Population<G>,
    fitness_fn: F,
    selection: SelectionOperator,
    crossover: CrossoverOperator<G>,
    mutation: MutationOperator<G>,
    generation: usize,
    fitness_history: Vec<f64>,
    cached_lower: Vec<f64>,
    cached_upper: Vec<f64>,
}

impl<G, F> GeneticAlgorithm<G, F>
where
    G: Genome + Clone + Send + Sync + Default,
    F: FitnessFunction<G> + Sync,
    SelectionOperator: Selection<G>,
    CrossoverOperator<G>: Crossover<G>,
    MutationOperator<G>: Mutation<G>,
{
    /// Creates a new genetic algorithm instance.
    pub fn new(config: GaConfig, fitness_fn: F) -> Self {
        let mut rng = rng::create_rng(config.seed);
        let (lower, upper) = config.bounds();
        let population = Population::random(&mut rng, config.population_size, &lower, &upper);

        Self {
            selection: SelectionOperator::Tournament(config.tournament_size),
            crossover: CrossoverOperator::default(),
            mutation: MutationOperator::default(),
            cached_lower: lower,
            cached_upper: upper,
            config,
            population,
            fitness_fn,
            generation: 0,
            fitness_history: Vec::new(),
        }
    }

    /// Creates a genetic algorithm with custom operators.
    pub fn with_operators(
        config: GaConfig,
        fitness_fn: F,
        population: Population<G>,
        selection: SelectionOperator,
        crossover: CrossoverOperator<G>,
        mutation: MutationOperator<G>,
    ) -> Self {
        let (lower, upper) = config.bounds();
        Self {
            config,
            population,
            fitness_fn,
            selection,
            crossover,
            mutation,
            generation: 0,
            fitness_history: Vec::new(),
            cached_lower: lower,
            cached_upper: upper,
        }
    }

    /// Sets the selection operator.
    pub fn with_selection(mut self, selection: SelectionOperator) -> Self {
        self.selection = selection;
        self
    }

    /// Sets the crossover operator.
    pub fn with_crossover(mut self, crossover: CrossoverOperator<G>) -> Self {
        self.crossover = crossover;
        self
    }

    /// Sets the mutation operator.
    pub fn with_mutation(mut self, mutation: MutationOperator<G>) -> Self {
        self.mutation = mutation;
        self
    }

    /// Runs the genetic algorithm for the configured number of generations.
    pub fn run(&mut self) -> GaResult<G> {
        // Evaluate initial population
        self.evaluate_population();

        let early_stopping = self.config.early_stopping;
        let restart_patience = self.config.restart_on_stagnation;
        let mutation_rate_start = self.config.mutation_rate;
        let mutation_rate_end = self.config.mutation_rate_end;
        let total_gens = self.config.generations;
        let mut stagnation_count: usize = 0;
        let mut best_so_far = f64::NEG_INFINITY;

        for gen in 0..total_gens {
            // Adaptive mutation rate: linear interpolation
            if let Some(end_rate) = mutation_rate_end {
                let t = gen as f64 / total_gens.max(1) as f64;
                self.config.mutation_rate =
                    mutation_rate_start + t * (end_rate - mutation_rate_start);
            }

            self.step();

            let current_best = self
                .fitness_history
                .last()
                .copied()
                .unwrap_or(f64::NEG_INFINITY);

            if current_best > best_so_far + 1e-12 {
                best_so_far = current_best;
                stagnation_count = 0;
            } else {
                stagnation_count += 1;
            }

            // Restart: reinitialize non-elite population on stagnation
            if let Some(patience) = restart_patience {
                if stagnation_count >= patience && stagnation_count % patience == 0 {
                    let mut restart_rng = rand::thread_rng();
                    let elitism = self.config.elitism;
                    let genome_length = self.config.genome_length;
                    let lower = &self.cached_lower;
                    let upper = &self.cached_upper;
                    let individuals = self.population.individuals_mut();
                    for ind in individuals.iter_mut().skip(elitism) {
                        *ind = Individual::new(G::random(
                            &mut restart_rng,
                            genome_length,
                            lower,
                            upper,
                        ));
                    }
                    self.evaluate_population();
                }
            }

            // Early stopping check (only if no restart is configured)
            if restart_patience.is_none() {
                if let Some(patience) = early_stopping {
                    if stagnation_count >= patience {
                        break;
                    }
                }
            }
        }

        self.result()
    }

    /// Runs a single generation step.
    pub fn step(&mut self) {
        let mut rng = rand::thread_rng();

        // Selection
        let parents = self.selection.select(
            &self.population,
            self.config.population_size - self.config.elitism,
            &mut rng,
        );

        // Create new population with elites
        let mut new_individuals: Vec<Individual<G>> =
            Vec::with_capacity(self.config.population_size);
        new_individuals.extend(
            self.population
                .individuals()
                .iter()
                .take(self.config.elitism)
                .cloned(),
        );

        // Crossover and mutation using cached bounds
        let lower = &self.cached_lower;
        let upper = &self.cached_upper;
        for chunk in parents.chunks(2) {
            if chunk.len() == 2 {
                let (child1, child2) = if rng.gen::<f64>() < self.config.crossover_rate {
                    self.crossover
                        .crossover(&chunk[0].genome, &chunk[1].genome, &mut rng)
                } else {
                    (chunk[0].genome.clone(), chunk[1].genome.clone())
                };

                let mut ind1 = Individual::new(child1);
                let mut ind2 = Individual::new(child2);

                self.mutation.mutate(
                    &mut ind1.genome,
                    self.config.mutation_rate,
                    lower,
                    upper,
                    &mut rng,
                );
                self.mutation.mutate(
                    &mut ind2.genome,
                    self.config.mutation_rate,
                    lower,
                    upper,
                    &mut rng,
                );

                new_individuals.push(ind1);
                if new_individuals.len() < self.config.population_size {
                    new_individuals.push(ind2);
                }
            } else if !chunk.is_empty() {
                let mut ind = chunk[0].clone();
                self.mutation.mutate(
                    &mut ind.genome,
                    self.config.mutation_rate,
                    lower,
                    upper,
                    &mut rng,
                );
                new_individuals.push(ind);
            }
        }

        self.population = Population::from_individuals(new_individuals);
        self.evaluate_population();

        // Random immigrants: replace worst individuals with random ones
        if let Some(n_immigrants) = self.config.random_immigrants {
            if n_immigrants > 0 {
                let pop_size = self.population.len();
                let mut imm_rng = rand::thread_rng();
                let genome_length = self.config.genome_length;
                let lower = &self.cached_lower;
                let upper = &self.cached_upper;
                let start = pop_size.saturating_sub(n_immigrants);
                let individuals = self.population.individuals_mut();
                for ind in individuals.iter_mut().skip(start) {
                    *ind = Individual::new(G::random(&mut imm_rng, genome_length, lower, upper));
                }
                // Evaluate the new immigrants
                let immigrant_genomes: Vec<&G> = self
                    .population
                    .individuals()
                    .iter()
                    .skip(start)
                    .map(|ind| &ind.genome)
                    .collect();
                let results = self.fitness_fn.evaluate_batch(&immigrant_genomes);
                for (i, fitness) in results.into_iter().enumerate() {
                    self.population.individuals_mut()[start + i].fitness = Some(fitness);
                }
                self.population.sort_by_fitness();
            }
        }

        // Local search on the best individual
        if let Some(iters) = self.config.local_search_iters {
            if iters > 0 {
                self.local_search(iters);
            }
        }

        self.generation += 1;
    }

    /// Apply local search (hill climbing) to the best individual.
    /// Tries small random perturbations and keeps improvements.
    fn local_search(&mut self, iterations: usize) {
        let best = match self.population.best() {
            Some(b) => b.clone(),
            None => return,
        };
        let best_fitness = best.fitness.unwrap_or(f64::NEG_INFINITY);
        let mut current_genes: Vec<f64> = best.genome.as_f64_vec();
        let mut current_fitness = best_fitness;

        let mut rng = rand::thread_rng();
        let lower = &self.cached_lower;
        let upper = &self.cached_upper;
        let len = current_genes.len();

        for _ in 0..iterations {
            // Perturb a random gene
            let idx = rng.gen_range(0..len);
            let range = upper[idx] - lower[idx];
            let sigma = range * 0.01; // 1% of range
            let delta: f64 = rng.gen_range(-sigma..=sigma);
            let old_val = current_genes[idx];
            current_genes[idx] = (old_val + delta).clamp(lower[idx], upper[idx]);

            let candidate = G::from_f64_vec(current_genes.clone());
            let fitness = self.fitness_fn.evaluate(&candidate);

            if fitness > current_fitness {
                current_fitness = fitness;
            } else {
                current_genes[idx] = old_val; // revert
            }
        }

        if current_fitness > best_fitness {
            let mut improved = Individual::new(G::from_f64_vec(current_genes));
            improved.fitness = Some(current_fitness);
            // Replace the best individual
            self.population.individuals_mut()[0] = improved;
            // Update fitness history
            if let Some(last) = self.fitness_history.last_mut() {
                *last = current_fitness;
            }
        }
    }

    /// Evaluates fitness for all individuals in the population.
    fn evaluate_population(&mut self) {
        // Collect indices that need evaluation
        let unevaluated: Vec<usize> = self
            .population
            .individuals()
            .iter()
            .enumerate()
            .filter(|(_, ind)| ind.fitness.is_none())
            .map(|(i, _)| i)
            .collect();

        if !unevaluated.is_empty() {
            // Collect genome references for batch evaluation
            let genomes: Vec<&G> = unevaluated
                .iter()
                .map(|&i| &self.population.individuals()[i].genome)
                .collect();

            let results = self.fitness_fn.evaluate_batch(&genomes);

            // Assign results back
            for (idx, fitness) in unevaluated.into_iter().zip(results) {
                self.population.individuals_mut()[idx].fitness = Some(fitness);
            }
        }

        // Sort by fitness (descending - higher is better)
        self.population.sort_by_fitness();

        // Record best fitness
        if let Some(best) = self.population.best() {
            self.fitness_history
                .push(best.fitness.unwrap_or(f64::NEG_INFINITY));
        }
    }

    /// Returns the current best individual.
    pub fn best(&self) -> Option<&Individual<G>> {
        self.population.best()
    }

    /// Returns the current generation number.
    pub fn generation(&self) -> usize {
        self.generation
    }

    /// Builds the final result.
    fn result(&self) -> GaResult<G> {
        let best = self
            .population
            .best()
            .cloned()
            .unwrap_or_else(|| Individual::new(G::default()));
        let best_fitness = best.fitness.unwrap_or(f64::NEG_INFINITY);

        // Check for convergence (fitness hasn't improved significantly in last 10 generations)
        let converged = if self.fitness_history.len() >= 10 {
            let recent: Vec<_> = self.fitness_history.iter().rev().take(10).collect();
            let variance = statistical_variance(recent.iter().copied().copied());
            variance < 1e-10
        } else {
            false
        };

        GaResult {
            best_individual: best,
            best_fitness,
            generations: self.generation,
            fitness_history: self.fitness_history.clone(),
            converged,
        }
    }
}

fn statistical_variance<I: Iterator<Item = f64>>(iter: I) -> f64 {
    let values: Vec<f64> = iter.collect();
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64
}

/// Prelude module for convenient imports.
pub mod prelude {
    pub use crate::{
        fitness::{benchmarks::*, FitnessFunction},
        genome::{BinaryGenome, Genome, PermutationGenome, RealGenome},
        island::{IslandConfig, IslandModel, MigrationTopology},
        operators::{
            crossover::{Crossover, CrossoverOperator},
            mutation::{Mutation, MutationOperator},
            selection::{Selection, SelectionOperator},
        },
        population::{Individual, Population, PopulationConfig},
        Error, GaConfig, GaConfigBuilder, GaResult, GeneticAlgorithm, Result,
    };
}

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Python module initialization.
#[cfg(feature = "python")]
#[pymodule(gil_used = false)]
fn _parga(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register_module(m)?;
    Ok(())
}
