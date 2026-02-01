//! Island model for parallel genetic algorithms.
//!
//! The island model maintains multiple independent populations (islands) that
//! evolve in parallel with occasional migration of individuals between islands.
//! This approach provides:
//!
//! - Better exploration of the search space
//! - Reduced genetic drift
//! - Natural parallelization
//! - Improved diversity maintenance

use crate::fitness::FitnessFunction;
use crate::genome::Genome;
use crate::operators::{
    Crossover, CrossoverOperator, Mutation, MutationOperator, Selection, SelectionOperator,
};
use crate::population::{Individual, Population};
use crate::rng;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use serde::{Deserialize, Serialize};

/// Configuration for the island model.
#[derive(Debug, Clone, derive_builder::Builder)]
pub struct IslandConfig {
    /// Number of islands.
    #[builder(default = "4")]
    pub num_islands: usize,

    /// Population size per island.
    #[builder(default = "100")]
    pub island_population: usize,

    /// Generations between migrations.
    #[builder(default = "10")]
    pub migration_interval: usize,

    /// Number of individuals to migrate.
    #[builder(default = "5")]
    pub migration_count: usize,

    /// Migration topology.
    #[builder(default = "MigrationTopology::Ring")]
    pub topology: MigrationTopology,

    /// Total generations to evolve.
    #[builder(default = "100")]
    pub generations: usize,

    /// Genome length.
    pub genome_length: usize,

    /// Mutation rate.
    #[builder(default = "0.01")]
    pub mutation_rate: f64,

    /// Crossover rate.
    #[builder(default = "0.8")]
    pub crossover_rate: f64,

    /// Number of elite individuals per island.
    #[builder(default = "2")]
    pub elitism: usize,

    /// Tournament size for selection.
    #[builder(default = "3")]
    pub tournament_size: usize,

    /// Lower bounds for genes.
    #[builder(default = "None", setter(strip_option))]
    pub lower_bounds: Option<Vec<f64>>,

    /// Upper bounds for genes.
    #[builder(default = "None", setter(strip_option))]
    pub upper_bounds: Option<Vec<f64>>,

    /// Random seed for reproducibility.
    #[builder(default = "None", setter(strip_option))]
    pub seed: Option<u64>,
}

impl IslandConfig {
    /// Creates a new builder.
    pub fn builder() -> IslandConfigBuilder {
        IslandConfigBuilder::default()
    }

    /// Returns bounds with defaults.
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

/// Migration topology between islands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MigrationTopology {
    /// Ring topology: island i sends to island (i+1) % n.
    Ring,

    /// Fully connected: every island can send to every other.
    FullyConnected,

    /// Random: randomly select destination islands.
    Random,

    /// Star: central island exchanges with all others.
    Star,

    /// Ladder: bidirectional between adjacent islands.
    Ladder,
}

impl MigrationTopology {
    /// Returns the destination islands for migration from a given island.
    pub fn destinations(&self, from: usize, num_islands: usize) -> Vec<usize> {
        match self {
            Self::Ring => vec![(from + 1) % num_islands],

            Self::FullyConnected => (0..num_islands).filter(|&i| i != from).collect(),

            Self::Random => {
                // Handled during migration
                vec![]
            }

            Self::Star => {
                if from == 0 {
                    (1..num_islands).collect()
                } else {
                    vec![0]
                }
            }

            Self::Ladder => {
                let mut dests = Vec::new();
                if from > 0 {
                    dests.push(from - 1);
                }
                if from < num_islands - 1 {
                    dests.push(from + 1);
                }
                dests
            }
        }
    }
}

/// Result of island model evolution.
#[derive(Debug, Clone)]
pub struct IslandResult<G: Genome> {
    /// Best individual found across all islands.
    pub best_individual: Individual<G>,

    /// Best fitness value.
    pub best_fitness: f64,

    /// Number of generations evolved.
    pub generations: usize,

    /// Best fitness per island at the end.
    pub island_best_fitness: Vec<f64>,

    /// Global best fitness history.
    pub fitness_history: Vec<f64>,

    /// Whether any island converged.
    pub converged: bool,
}

/// Island model genetic algorithm.
pub struct IslandModel<G, F>
where
    G: Genome,
    F: FitnessFunction<G>,
{
    config: IslandConfig,
    islands: Vec<Population<G>>,
    fitness_fn: F,
    selection: SelectionOperator,
    crossover: CrossoverOperator<G>,
    mutation: MutationOperator<G>,
    generation: usize,
    fitness_history: Vec<f64>,
}

impl<G, F> IslandModel<G, F>
where
    G: Genome + Clone + Send + Sync + Default,
    F: FitnessFunction<G> + Clone + Send + Sync,
    SelectionOperator: Selection<G>,
    CrossoverOperator<G>: Crossover<G>,
    MutationOperator<G>: Mutation<G>,
{
    /// Creates a new island model.
    pub fn new(config: IslandConfig, fitness_fn: F) -> Self {
        let mut rng = rng::create_rng(config.seed);
        let (lower, upper) = config.bounds();

        let islands: Vec<Population<G>> = (0..config.num_islands)
            .map(|_| Population::random(&mut rng, config.island_population, &lower, &upper))
            .collect();

        Self {
            selection: SelectionOperator::Tournament(config.tournament_size),
            crossover: CrossoverOperator::default(),
            mutation: MutationOperator::default(),
            config,
            islands,
            fitness_fn,
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

    /// Runs the island model evolution.
    pub fn run(&mut self) -> IslandResult<G> {
        // Initial evaluation
        self.evaluate_all_islands();
        self.record_best_fitness();

        for gen in 0..self.config.generations {
            self.generation = gen;

            // Evolve each island
            self.evolve_islands();

            // Evaluate fitness
            self.evaluate_all_islands();

            // Record best
            self.record_best_fitness();

            // Migrate if interval reached
            if (gen + 1) % self.config.migration_interval == 0 {
                self.migrate();
            }
        }

        self.result()
    }

    /// Runs a single generation on all islands.
    pub fn step(&mut self) {
        self.evolve_islands();
        self.evaluate_all_islands();
        self.record_best_fitness();

        if (self.generation + 1) % self.config.migration_interval == 0 {
            self.migrate();
        }

        self.generation += 1;
    }

    /// Evolves all islands for one generation.
    fn evolve_islands(&mut self) {
        let (lower, upper) = self.config.bounds();
        let config = &self.config;
        let selection = &self.selection;
        let crossover = &self.crossover;
        let mutation = &self.mutation;

        #[cfg(feature = "parallel")]
        {
            self.islands.par_iter_mut().for_each(|island| {
                evolve_population(
                    island, config, selection, crossover, mutation, &lower, &upper,
                );
            });
        }

        #[cfg(not(feature = "parallel"))]
        {
            for island in &mut self.islands {
                evolve_population(
                    island, config, selection, crossover, mutation, &lower, &upper,
                );
            }
        }
    }

    /// Evaluates fitness for all islands.
    fn evaluate_all_islands(&mut self) {
        let fitness_fn = &self.fitness_fn;

        #[cfg(feature = "parallel")]
        {
            self.islands.par_iter_mut().for_each(|island| {
                for ind in island.individuals_mut() {
                    if ind.fitness.is_none() {
                        ind.fitness = Some(fitness_fn.evaluate(&ind.genome));
                    }
                }
                island.sort_by_fitness();
            });
        }

        #[cfg(not(feature = "parallel"))]
        {
            for island in &mut self.islands {
                for ind in island.individuals_mut() {
                    if ind.fitness.is_none() {
                        ind.fitness = Some(fitness_fn.evaluate(&ind.genome));
                    }
                }
                island.sort_by_fitness();
            }
        }
    }

    /// Records the best fitness across all islands.
    fn record_best_fitness(&mut self) {
        let best = self
            .islands
            .iter()
            .filter_map(|island| island.best())
            .max_by(|a, b| {
                a.fitness
                    .partial_cmp(&b.fitness)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        if let Some(best) = best {
            self.fitness_history
                .push(best.fitness.unwrap_or(f64::NEG_INFINITY));
        }
    }

    /// Performs migration between islands.
    #[allow(clippy::needless_range_loop)]
    fn migrate(&mut self) {
        let num_islands = self.config.num_islands;
        let migration_count = self.config.migration_count;
        let topology = self.config.topology;

        match topology {
            MigrationTopology::Ring => {
                // Ring migration: island i sends to island (i+1) % n
                let mut migrants: Vec<Vec<Individual<G>>> = Vec::with_capacity(num_islands);

                // Collect migrants from each island (best individuals)
                for island in &self.islands {
                    let best: Vec<_> = island
                        .best_n(migration_count)
                        .into_iter()
                        .cloned()
                        .collect();
                    migrants.push(best);
                }

                // Send migrants to next island
                for (i, island_migrants) in migrants.into_iter().enumerate() {
                    let dest = (i + 1) % num_islands;
                    for mut migrant in island_migrants {
                        migrant.invalidate_fitness(); // Re-evaluate in new context
                        self.islands[dest].push(migrant);
                    }
                    // Trim destination island to maintain size
                    self.islands[dest].truncate(self.config.island_population);
                }
            }

            MigrationTopology::FullyConnected => {
                // Each island sends to all others
                let mut all_migrants: Vec<Vec<Individual<G>>> = Vec::with_capacity(num_islands);

                for island in &self.islands {
                    all_migrants.push(
                        island
                            .best_n(migration_count)
                            .into_iter()
                            .cloned()
                            .collect(),
                    );
                }

                for from in 0..num_islands {
                    for to in 0..num_islands {
                        if from != to {
                            // Send one migrant to each destination
                            if let Some(migrant) =
                                all_migrants[from].get(to % all_migrants[from].len())
                            {
                                let mut m = migrant.clone();
                                m.invalidate_fitness();
                                self.islands[to].push(m);
                            }
                        }
                    }
                }

                for island in &mut self.islands {
                    island.truncate(self.config.island_population);
                }
            }

            MigrationTopology::Random => {
                let mut rng = rng::create_rng(None);
                let mut migrants: Vec<Vec<Individual<G>>> = Vec::with_capacity(num_islands);

                for island in &self.islands {
                    migrants.push(
                        island
                            .best_n(migration_count)
                            .into_iter()
                            .cloned()
                            .collect(),
                    );
                }

                for (from, island_migrants) in migrants.into_iter().enumerate() {
                    for mut migrant in island_migrants {
                        use rand::Rng;
                        let mut dest = rng.gen_range(0..num_islands);
                        while dest == from {
                            dest = rng.gen_range(0..num_islands);
                        }
                        migrant.invalidate_fitness();
                        self.islands[dest].push(migrant);
                    }
                }

                for island in &mut self.islands {
                    island.truncate(self.config.island_population);
                }
            }

            MigrationTopology::Star => {
                // Island 0 is the hub
                let hub_migrants: Vec<Individual<G>> = self.islands[0]
                    .best_n(migration_count)
                    .into_iter()
                    .cloned()
                    .collect();

                // Collect from satellites
                let mut satellite_migrants: Vec<Vec<Individual<G>>> = Vec::new();
                for i in 1..num_islands {
                    satellite_migrants.push(
                        self.islands[i]
                            .best_n(migration_count)
                            .into_iter()
                            .cloned()
                            .collect(),
                    );
                }

                // Hub sends to satellites
                for i in 1..num_islands {
                    if let Some(mut migrant) =
                        hub_migrants.get((i - 1) % hub_migrants.len()).cloned()
                    {
                        migrant.invalidate_fitness();
                        self.islands[i].push(migrant);
                    }
                    self.islands[i].truncate(self.config.island_population);
                }

                // Satellites send to hub
                for migrants in satellite_migrants {
                    for mut migrant in migrants {
                        migrant.invalidate_fitness();
                        self.islands[0].push(migrant);
                    }
                }
                self.islands[0].truncate(self.config.island_population);
            }

            MigrationTopology::Ladder => {
                let mut migrants: Vec<Vec<Individual<G>>> = Vec::with_capacity(num_islands);

                for island in &self.islands {
                    migrants.push(
                        island
                            .best_n(migration_count)
                            .into_iter()
                            .cloned()
                            .collect(),
                    );
                }

                for i in 0..num_islands {
                    // Send to previous
                    if i > 0 {
                        if let Some(migrant) = migrants[i].first() {
                            let mut m = migrant.clone();
                            m.invalidate_fitness();
                            self.islands[i - 1].push(m);
                        }
                    }
                    // Send to next
                    if i < num_islands - 1 {
                        if let Some(migrant) = migrants[i].last() {
                            let mut m = migrant.clone();
                            m.invalidate_fitness();
                            self.islands[i + 1].push(m);
                        }
                    }
                }

                for island in &mut self.islands {
                    island.truncate(self.config.island_population);
                }
            }
        }
    }

    /// Returns the global best individual.
    pub fn best(&self) -> Option<&Individual<G>> {
        self.islands
            .iter()
            .filter_map(|island| island.best())
            .max_by(|a, b| {
                a.fitness
                    .partial_cmp(&b.fitness)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Returns the current generation.
    pub fn generation(&self) -> usize {
        self.generation
    }

    /// Returns references to all islands.
    pub fn islands(&self) -> &[Population<G>] {
        &self.islands
    }

    /// Builds the final result.
    fn result(&self) -> IslandResult<G> {
        let best = self
            .best()
            .cloned()
            .unwrap_or_else(|| Individual::new(G::default()));

        let best_fitness = best.fitness.unwrap_or(f64::NEG_INFINITY);

        let island_best_fitness: Vec<f64> = self
            .islands
            .iter()
            .map(|island| {
                island
                    .best()
                    .and_then(|i| i.fitness)
                    .unwrap_or(f64::NEG_INFINITY)
            })
            .collect();

        // Check convergence
        let converged = if self.fitness_history.len() >= 10 {
            let recent: Vec<_> = self
                .fitness_history
                .iter()
                .rev()
                .take(10)
                .copied()
                .collect();
            let mean = recent.iter().sum::<f64>() / recent.len() as f64;
            let variance =
                recent.iter().map(|f| (f - mean).powi(2)).sum::<f64>() / recent.len() as f64;
            variance < 1e-10
        } else {
            false
        };

        IslandResult {
            best_individual: best,
            best_fitness,
            generations: self.generation,
            island_best_fitness,
            fitness_history: self.fitness_history.clone(),
            converged,
        }
    }
}

/// Helper function to evolve a single population.
fn evolve_population<G>(
    population: &mut Population<G>,
    config: &IslandConfig,
    selection: &SelectionOperator,
    crossover: &CrossoverOperator<G>,
    mutation: &MutationOperator<G>,
    lower: &[f64],
    upper: &[f64],
) where
    G: Genome + Clone,
    SelectionOperator: Selection<G>,
    CrossoverOperator<G>: Crossover<G>,
    MutationOperator<G>: Mutation<G>,
{
    let mut rng = rng::create_rng(None);

    // Selection
    let parents = selection.select(
        population,
        config.island_population - config.elitism,
        &mut rng,
    );

    // Preserve elites
    population.sort_by_fitness();
    let mut new_individuals: Vec<Individual<G>> = population
        .individuals()
        .iter()
        .take(config.elitism)
        .cloned()
        .collect();

    // Crossover and mutation
    for chunk in parents.chunks(2) {
        if chunk.len() == 2 {
            let (child1, child2) = if rand::random::<f64>() < config.crossover_rate {
                crossover.crossover(&chunk[0].genome, &chunk[1].genome, &mut rng)
            } else {
                (chunk[0].genome.clone(), chunk[1].genome.clone())
            };

            let mut ind1 = Individual::new(child1);
            let mut ind2 = Individual::new(child2);

            mutation.mutate(
                &mut ind1.genome,
                config.mutation_rate,
                lower,
                upper,
                &mut rng,
            );
            mutation.mutate(
                &mut ind2.genome,
                config.mutation_rate,
                lower,
                upper,
                &mut rng,
            );

            new_individuals.push(ind1);
            if new_individuals.len() < config.island_population {
                new_individuals.push(ind2);
            }
        } else if !chunk.is_empty() {
            let mut ind = chunk[0].clone();
            mutation.mutate(
                &mut ind.genome,
                config.mutation_rate,
                lower,
                upper,
                &mut rng,
            );
            new_individuals.push(ind);
        }
    }

    *population = Population::from_individuals(new_individuals);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitness::benchmarks::Sphere;
    use crate::genome::RealGenome;

    #[test]
    fn test_island_config() {
        let config = IslandConfig::builder()
            .num_islands(4)
            .island_population(50)
            .genome_length(10)
            .build()
            .unwrap();

        assert_eq!(config.num_islands, 4);
        assert_eq!(config.island_population, 50);
    }

    #[test]
    fn test_migration_topology_ring() {
        let dests = MigrationTopology::Ring.destinations(0, 4);
        assert_eq!(dests, vec![1]);

        let dests = MigrationTopology::Ring.destinations(3, 4);
        assert_eq!(dests, vec![0]);
    }

    #[test]
    fn test_island_model_basic() {
        let config = IslandConfig::builder()
            .num_islands(2)
            .island_population(20)
            .genome_length(5)
            .generations(10)
            .migration_interval(5)
            .build()
            .unwrap();

        let fitness = Sphere;
        let mut island_model: IslandModel<RealGenome, _> = IslandModel::new(config, fitness);

        let result = island_model.run();

        assert!(result.best_fitness.is_finite());
        assert_eq!(result.island_best_fitness.len(), 2);
    }
}
