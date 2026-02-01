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
        Self {
            config,
            population,
            fitness_fn,
            selection,
            crossover,
            mutation,
            generation: 0,
            fitness_history: Vec::new(),
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

        for _ in 0..self.config.generations {
            self.step();
        }

        self.result()
    }

    /// Runs a single generation step.
    pub fn step(&mut self) {
        let mut rng = rng::create_rng(None);

        // Selection
        let parents = self.selection.select(
            &self.population,
            self.config.population_size - self.config.elitism,
            &mut rng,
        );

        // Create new population with elites
        let mut new_individuals: Vec<Individual<G>> = self
            .population
            .individuals()
            .iter()
            .take(self.config.elitism)
            .cloned()
            .collect();

        // Crossover and mutation
        let (lower, upper) = self.config.bounds();
        for chunk in parents.chunks(2) {
            if chunk.len() == 2 {
                let (child1, child2) = if rand::random::<f64>() < self.config.crossover_rate {
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
                    &lower,
                    &upper,
                    &mut rng,
                );
                self.mutation.mutate(
                    &mut ind2.genome,
                    self.config.mutation_rate,
                    &lower,
                    &upper,
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
                    &lower,
                    &upper,
                    &mut rng,
                );
                new_individuals.push(ind);
            }
        }

        self.population = Population::from_individuals(new_individuals);
        self.evaluate_population();
        self.generation += 1;
    }

    /// Evaluates fitness for all individuals in the population.
    fn evaluate_population(&mut self) {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            self.population
                .individuals_mut()
                .par_iter_mut()
                .for_each(|ind| {
                    if ind.fitness.is_none() {
                        ind.fitness = Some(self.fitness_fn.evaluate(&ind.genome));
                    }
                });
        }

        #[cfg(not(feature = "parallel"))]
        {
            for ind in self.population.individuals_mut() {
                if ind.fitness.is_none() {
                    ind.fitness = Some(self.fitness_fn.evaluate(&ind.genome));
                }
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
#[pymodule]
fn _parga(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register_module(m)?;
    Ok(())
}
