//! Integration tests for the parga library.

use parga::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use parga::{
    fitness::fitness_fn,
    island::{IslandConfigBuilder, IslandResult},
    operators::crossover::{CrossoverOperator, PermutationCrossover, RealCrossover},
    operators::mutation::{MutationOperator, PermutationMutation},
};

fn assert_valid_result(result: &GaResult<RealGenome>) {
    assert!(result.best_fitness.is_finite());
    assert!(!result.fitness_history.is_empty());
    assert!(
        result
            .fitness_history
            .windows(2)
            .all(|pair| pair[1] >= pair[0]),
        "elitism should preserve the best fitness"
    );
}

#[test]
fn test_simple_optimization() {
    let config = GaConfig::builder()
        .population_size(50)
        .genome_length(5)
        .generations(50)
        .mutation_rate(0.1)
        .seed(42)
        .build()
        .unwrap();

    let fitness = Sphere;
    let mut ga: GeneticAlgorithm<RealGenome, _> = GeneticAlgorithm::new(config, fitness).unwrap();
    let result = ga.run();

    // Should find a solution close to zero
    assert!(result.best_fitness > -1.0, "Fitness should be close to 0");
}

#[test]
fn test_rastrigin_optimization() {
    let config = GaConfig::builder()
        .population_size(100)
        .genome_length(3)
        .generations(100)
        .mutation_rate(0.05)
        .lower_bounds(vec![-5.12; 3])
        .upper_bounds(vec![5.12; 3])
        .seed(42)
        .build()
        .unwrap();

    let fitness = Rastrigin;
    let mut ga: GeneticAlgorithm<RealGenome, _> = GeneticAlgorithm::new(config, fitness).unwrap();
    let result = ga.run();

    // Should find a reasonable solution (Rastrigin is harder)
    assert!(result.best_fitness > -10.0);
}

#[test]
fn test_real_crossover_offspring_stay_within_bounds() {
    let crossovers = [
        RealCrossover::Blend(0.5),
        RealCrossover::SimulatedBinary(2.0),
    ];

    for crossover in crossovers {
        let config = GaConfig::builder()
            .population_size(50)
            .genome_length(4)
            .generations(30)
            .mutation_rate(0.0)
            .crossover_rate(1.0)
            .elitism(0)
            .lower_bounds(vec![-0.01; 4])
            .upper_bounds(vec![0.01; 4])
            .seed(42)
            .build()
            .unwrap();

        let fitness =
            fitness_fn(|genome: &RealGenome| genome.genes().iter().map(|gene| gene.abs()).sum());
        let mut ga = GeneticAlgorithm::new(config, fitness)
            .unwrap()
            .with_crossover(CrossoverOperator::Real(crossover));
        let result = ga.run();

        assert!(
            result
                .best_individual
                .genome
                .genes()
                .iter()
                .all(|gene| (-0.01..=0.01).contains(gene)),
            "{crossover:?} produced an out-of-bounds best genome: {:?}",
            result.best_individual.genome.genes()
        );
    }
}

#[test]
fn test_island_real_crossover_offspring_stay_within_bounds() {
    let config = IslandConfig::builder()
        .num_islands(2)
        .island_population(30)
        .genome_length(4)
        .generations(20)
        .mutation_rate(0.0)
        .crossover_rate(1.0)
        .elitism(0)
        .lower_bounds(vec![-0.01; 4])
        .upper_bounds(vec![0.01; 4])
        .migration_interval(20)
        .seed(42)
        .build()
        .unwrap();

    let fitness =
        fitness_fn(|genome: &RealGenome| genome.genes().iter().map(|gene| gene.abs()).sum());
    let mut model = IslandModel::new(config, fitness)
        .unwrap()
        .with_crossover(CrossoverOperator::Real(RealCrossover::Blend(0.5)));
    let result = model.run();

    assert!(
        result
            .best_individual
            .genome
            .genes()
            .iter()
            .all(|gene| (-0.01..=0.01).contains(gene)),
        "island crossover produced an out-of-bounds best genome: {:?}",
        result.best_individual.genome.genes()
    );
}

#[test]
fn test_island_model_basic() {
    let config = IslandConfig::builder()
        .num_islands(2)
        .island_population(30)
        .genome_length(5)
        .generations(30)
        .migration_interval(10)
        .migration_count(3)
        .seed(42)
        .build()
        .unwrap();

    let fitness = Sphere;
    let mut island_model: IslandModel<RealGenome, _> = IslandModel::new(config, fitness).unwrap();
    let result = island_model.run();

    // Island model should find a reasonable solution for Sphere
    assert!(result.best_fitness > -5.0);
    assert_eq!(result.island_best_fitness.len(), 2);
}

#[test]
fn test_island_model_topologies() {
    let topologies = [
        MigrationTopology::Ring,
        MigrationTopology::Star,
        MigrationTopology::Ladder,
        MigrationTopology::FullyConnected,
        MigrationTopology::Random,
    ];

    for topology in topologies {
        let config = IslandConfig::builder()
            .num_islands(3)
            .island_population(20)
            .genome_length(3)
            .generations(20)
            .migration_interval(5)
            .topology(topology)
            .seed(42)
            .build()
            .unwrap();

        let fitness = Sphere;
        let mut model: IslandModel<RealGenome, _> = IslandModel::new(config, fitness).unwrap();
        let result = model.run();

        assert!(
            result.best_fitness.is_finite(),
            "Topology {:?} failed",
            topology
        );
    }
}

#[test]
fn test_selection_methods() {
    let methods = [
        SelectionOperator::Tournament(3),
        SelectionOperator::RouletteWheel,
        SelectionOperator::Rank,
        SelectionOperator::Random,
        SelectionOperator::Truncation(0.5),
        SelectionOperator::StochasticUniversal,
    ];

    for method in methods {
        let config = GaConfig::builder()
            .population_size(30)
            .genome_length(3)
            .generations(20)
            .seed(42)
            .build()
            .unwrap();

        let fitness = Sphere;
        let mut ga: GeneticAlgorithm<RealGenome, _> = GeneticAlgorithm::new(config, fitness)
            .unwrap()
            .with_selection(method);
        let result = ga.run();

        assert!(
            result.best_fitness.is_finite(),
            "Selection {:?} failed",
            method
        );
    }
}

#[test]
fn test_invalid_ga_configurations_are_rejected_at_construction() {
    let cases = [
        GaConfig::builder()
            .population_size(2)
            .genome_length(1)
            .elitism(3)
            .build()
            .unwrap(),
        GaConfig::builder().genome_length(0).build().unwrap(),
        GaConfig::builder()
            .population_size(2)
            .genome_length(1)
            .tournament_size(3)
            .build()
            .unwrap(),
        GaConfig::builder()
            .genome_length(1)
            .mutation_rate(1.1)
            .build()
            .unwrap(),
        GaConfig::builder()
            .genome_length(2)
            .lower_bounds(vec![0.0])
            .upper_bounds(vec![1.0, 1.0])
            .build()
            .unwrap(),
        GaConfig::builder()
            .genome_length(1)
            .lower_bounds(vec![2.0])
            .upper_bounds(vec![1.0])
            .build()
            .unwrap(),
    ];

    for config in cases {
        let error = GeneticAlgorithm::<RealGenome, _>::new(config, Sphere)
            .err()
            .unwrap();
        assert!(error.to_string().starts_with("Configuration error:"));
    }
}

#[test]
fn test_invalid_island_configurations_are_rejected_at_construction() {
    let cases = [
        IslandConfig::builder()
            .num_islands(2)
            .genome_length(1)
            .migration_interval(0)
            .build()
            .unwrap(),
        IslandConfig::builder()
            .num_islands(1)
            .genome_length(1)
            .topology(MigrationTopology::Random)
            .build()
            .unwrap(),
        IslandConfig::builder()
            .num_islands(2)
            .island_population(2)
            .genome_length(1)
            .migration_count(3)
            .build()
            .unwrap(),
        IslandConfig::builder()
            .num_islands(2)
            .island_population(2)
            .genome_length(1)
            .elitism(3)
            .build()
            .unwrap(),
    ];

    for config in cases {
        let error = IslandModel::<RealGenome, _>::new(config, Sphere)
            .err()
            .unwrap();
        assert!(error.to_string().starts_with("Configuration error:"));
    }
}

#[test]
fn test_custom_fitness_function() {
    // Custom fitness: maximize x[0] + x[1] - x[2]
    let fitness = parga::fitness::fitness_fn(|genome: &RealGenome| {
        let genes = genome.genes();
        genes.first().unwrap_or(&0.0) + genes.get(1).unwrap_or(&0.0) - genes.get(2).unwrap_or(&0.0)
    });

    let config = GaConfig::builder()
        .population_size(50)
        .genome_length(3)
        .generations(50)
        .lower_bounds(vec![-10.0, -10.0, -10.0])
        .upper_bounds(vec![10.0, 10.0, 10.0])
        .seed(42)
        .build()
        .unwrap();

    let mut ga = GeneticAlgorithm::new(config, fitness).unwrap();
    let result = ga.run();

    // Best solution should have x[0], x[1] near 10 and x[2] near -10
    // Maximum fitness = 10 + 10 - (-10) = 30
    assert!(result.best_fitness > 20.0);
}

#[test]
fn test_convergence_detection() {
    let config = GaConfig::builder()
        .population_size(100)
        .genome_length(2)
        .generations(200)
        .mutation_rate(0.01)
        .seed(42)
        .build()
        .unwrap();

    let fitness = Sphere;
    let mut ga: GeneticAlgorithm<RealGenome, _> = GeneticAlgorithm::new(config, fitness).unwrap();
    let result = ga.run();

    // With enough generations, should converge
    // Note: convergence depends on the random seed
    assert!(!result.fitness_history.is_empty());
}

#[test]
fn test_early_stopping_shortens_run() {
    let configured_generations = 50;
    let config = GaConfig::builder()
        .population_size(10)
        .genome_length(3)
        .generations(configured_generations)
        .elitism(10)
        .early_stopping(1)
        .seed(42)
        .build()
        .unwrap();

    let mut ga: GeneticAlgorithm<RealGenome, _> = GeneticAlgorithm::new(config, Sphere).unwrap();
    let result = ga.run();

    assert_valid_result(&result);
    assert!(result.generations < configured_generations);
}

#[test]
fn test_restart_on_stagnation_completes_configured_run() {
    let configured_generations = 6;
    let config = GaConfig::builder()
        .population_size(20)
        .genome_length(3)
        .generations(configured_generations)
        .elitism(2)
        .mutation_rate(0.0)
        .crossover_rate(0.0)
        .early_stopping(1)
        .restart_on_stagnation(1)
        .seed(42)
        .build()
        .unwrap();

    let mut ga: GeneticAlgorithm<RealGenome, _> = GeneticAlgorithm::new(config, Sphere).unwrap();
    let result = ga.run();

    assert_valid_result(&result);
    assert_eq!(result.generations, configured_generations);
}

#[test]
fn test_restart_on_stagnation_records_one_history_entry_per_generation() {
    let configured_generations = 8;
    let config = GaConfig::builder()
        .population_size(20)
        .genome_length(3)
        .generations(configured_generations)
        .elitism(2)
        .mutation_rate(0.0)
        .crossover_rate(0.0)
        .restart_on_stagnation(1)
        .seed(42)
        .build()
        .unwrap();

    let mut ga: GeneticAlgorithm<RealGenome, _> = GeneticAlgorithm::new(config, Sphere).unwrap();
    let result = ga.run();

    assert_valid_result(&result);
    assert_eq!(
        result.fitness_history.len(),
        configured_generations + 1,
        "one entry per generation plus the initial evaluation"
    );
}

#[test]
fn test_local_search_completes_with_valid_history() {
    let config = GaConfig::builder()
        .population_size(20)
        .genome_length(3)
        .generations(6)
        .local_search_iters(5)
        .seed(42)
        .build()
        .unwrap();

    let mut ga: GeneticAlgorithm<RealGenome, _> = GeneticAlgorithm::new(config, Rastrigin).unwrap();
    let result = ga.run();

    assert_valid_result(&result);
}

#[test]
fn test_mutation_rate_decay_completes_with_valid_history() {
    let config = GaConfig::builder()
        .population_size(20)
        .genome_length(3)
        .generations(6)
        .mutation_rate(0.2)
        .mutation_rate_end(0.0)
        .seed(42)
        .build()
        .unwrap();

    let mut ga: GeneticAlgorithm<RealGenome, _> = GeneticAlgorithm::new(config, Sphere).unwrap();
    let result = ga.run();

    assert_valid_result(&result);
}

#[test]
fn test_random_immigrants_complete_with_valid_history() {
    let config = GaConfig::builder()
        .population_size(20)
        .genome_length(3)
        .generations(6)
        .random_immigrants(5)
        .seed(42)
        .build()
        .unwrap();

    let mut ga: GeneticAlgorithm<RealGenome, _> = GeneticAlgorithm::new(config, Rastrigin).unwrap();
    let result = ga.run();

    assert_valid_result(&result);
}

#[test]
fn test_reproducibility_with_seed() {
    let config1 = GaConfig::builder()
        .population_size(50)
        .genome_length(5)
        .generations(30)
        .seed(12345)
        .build()
        .unwrap();

    let config2 = config1.clone();

    let fitness = Sphere;

    let mut ga1: GeneticAlgorithm<RealGenome, _> = GeneticAlgorithm::new(config1, fitness).unwrap();
    let result1 = ga1.run();

    let mut ga2: GeneticAlgorithm<RealGenome, _> = GeneticAlgorithm::new(config2, fitness).unwrap();
    let result2 = ga2.run();

    // Results should be identical with the same seed
    assert!((result1.best_fitness - result2.best_fitness).abs() < 1e-10);
    assert_eq!(
        result1.best_individual.genome.genes(),
        result2.best_individual.genome.genes()
    );
}

#[test]
fn test_population_statistics() {
    let config = GaConfig::builder()
        .population_size(50)
        .genome_length(5)
        .generations(10)
        .seed(42)
        .build()
        .unwrap();

    let fitness = Sphere;
    let mut ga: GeneticAlgorithm<RealGenome, _> = GeneticAlgorithm::new(config, fitness).unwrap();
    let result = ga.run();

    assert!(!result.fitness_history.is_empty());
    assert!(result.fitness_history.len() <= 11); // generations + 1 for initial
}

#[test]
fn test_booth_function() {
    let config = GaConfig::builder()
        .population_size(100)
        .genome_length(2)
        .generations(100)
        .lower_bounds(vec![-10.0, -10.0])
        .upper_bounds(vec![10.0, 10.0])
        .seed(42)
        .build()
        .unwrap();

    let fitness = Booth;
    let mut ga: GeneticAlgorithm<RealGenome, _> = GeneticAlgorithm::new(config, fitness).unwrap();
    let result = ga.run();

    // Booth optimum is at (1, 3) with fitness 0
    // Should get reasonably close
    assert!(result.best_fitness > -5.0);
}

#[test]
fn test_permutation_ga_with_pmx_crossover() {
    let config = GaConfig::builder()
        .population_size(20)
        .genome_length(6)
        .generations(5)
        .mutation_rate(0.2)
        .seed(42)
        .build()
        .unwrap();

    // Reward orderings that are close to the identity permutation.
    let fitness = fitness_fn(|genome: &PermutationGenome| {
        let matches = genome
            .order()
            .iter()
            .enumerate()
            .filter(|(i, &v)| *i == v)
            .count();
        matches as f64
    });

    let mut ga: GeneticAlgorithm<PermutationGenome, _> = GeneticAlgorithm::new(config, fitness)
        .unwrap()
        .with_crossover(CrossoverOperator::Permutation(
            PermutationCrossover::PartiallyMapped,
        ))
        .with_mutation(MutationOperator::Permutation(PermutationMutation::Swap));

    let result = ga.run();

    assert_eq!(result.fitness_history.len(), 6);
    let mut sorted = result.best_individual.genome.order().to_vec();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..6).collect::<Vec<_>>());
}

#[test]
fn test_local_search_iters_leaves_permutation_genomes_valid() {
    let config = GaConfig::builder()
        .population_size(20)
        .genome_length(8)
        .generations(5)
        .mutation_rate(0.2)
        .local_search_iters(20)
        .seed(1)
        .build()
        .unwrap();

    // Reward orderings that are close to the identity permutation.
    let fitness = fitness_fn(|genome: &PermutationGenome| {
        let matches = genome
            .order()
            .iter()
            .enumerate()
            .filter(|(i, &v)| *i == v)
            .count();
        matches as f64
    });

    let mut ga: GeneticAlgorithm<PermutationGenome, _> = GeneticAlgorithm::new(config, fitness)
        .unwrap()
        .with_crossover(CrossoverOperator::Permutation(PermutationCrossover::Order))
        .with_mutation(MutationOperator::Permutation(PermutationMutation::Swap));

    let result = ga.run();

    assert_eq!(result.fitness_history.len(), 6);
    let order = result.best_individual.genome.order().to_vec();
    let mut sorted = order.clone();
    sorted.sort_unstable();
    assert_eq!(
        sorted,
        (0..8).collect::<Vec<_>>(),
        "local search must not corrupt a permutation genome: {order:?}"
    );
}

#[test]
fn test_local_search_iters_is_skipped_for_binary_genomes() {
    let config = GaConfig::builder()
        .population_size(20)
        .genome_length(8)
        .generations(5)
        .mutation_rate(0.2)
        .local_search_iters(20)
        .seed(1)
        .build()
        .unwrap();

    let evaluations = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&evaluations);
    let fitness = fitness_fn(move |genome: &BinaryGenome| {
        counter.fetch_add(1, Ordering::Relaxed);
        genome.count_ones() as f64
    });

    let mut ga: GeneticAlgorithm<BinaryGenome, _> = GeneticAlgorithm::new(config, fitness).unwrap();
    let result = ga.run();

    assert_eq!(result.fitness_history.len(), 6);
    // Evolution alone evaluates 110 genomes here; running local search would add
    // `generations * local_search_iters` = 100 more that can never flip a bit,
    // because the perturbation is far below the 0.5 threshold
    // `BinaryGenome::from_f64_vec` rounds at and non-improving steps are reverted.
    let evaluated = evaluations.load(Ordering::Relaxed);
    assert!(
        evaluated < 150,
        "local search should not evaluate anything for binary genomes, saw {evaluated} evaluations"
    );
}

fn island_config_builder(generations: usize) -> IslandConfigBuilder {
    let mut builder = IslandConfig::builder();
    builder
        .num_islands(2)
        .island_population(10)
        .genome_length(3)
        .generations(generations)
        .migration_interval(2)
        .migration_count(2)
        .seed(7);
    builder
}

#[test]
fn test_island_run_reports_generations_actually_evolved() {
    for generations in [3, 7] {
        let config = island_config_builder(generations).build().unwrap();
        let mut model: IslandModel<RealGenome, _> = IslandModel::new(config, Sphere).unwrap();
        let result = model.run();

        assert_eq!(result.generations, generations);
        assert_eq!(result.fitness_history.len(), generations + 1);
    }
}

#[test]
fn test_island_early_stopping_shortens_run() {
    let config = island_config_builder(10).early_stopping(1).build().unwrap();

    // A constant objective makes this deterministic: generation 0 always
    // improves on -inf, generation 1 always stagnates and trips patience = 1.
    let mut model: IslandModel<RealGenome, _> =
        IslandModel::new(config, fitness_fn(|_: &RealGenome| -1.0)).unwrap();
    let result = model.run();

    assert_eq!(result.generations, 2);
    assert_eq!(result.fitness_history.len(), result.generations + 1);
}

#[test]
fn test_island_without_early_stopping_runs_every_generation() {
    let config = island_config_builder(10).build().unwrap();

    let mut model: IslandModel<RealGenome, _> =
        IslandModel::new(config, fitness_fn(|_: &RealGenome| -1.0)).unwrap();
    let result = model.run();

    assert_eq!(result.generations, 10);
    assert_eq!(result.fitness_history.len(), 11);
}

fn seeded_island_run(mutation_rate_end: Option<f64>) -> IslandResult<RealGenome> {
    let mut builder = island_config_builder(6);
    builder.mutation_rate(0.5);
    if let Some(end) = mutation_rate_end {
        builder.mutation_rate_end(end);
    }
    let config = builder.build().unwrap();

    let mut model: IslandModel<RealGenome, _> = IslandModel::new(config, Sphere).unwrap();
    model.run()
}

#[test]
fn test_island_mutation_rate_end_identity_reproduces_undecayed_run() {
    let baseline = seeded_island_run(None);
    // A final rate equal to the starting rate must consume the same RNG stream
    // and produce a bit-identical run.
    let flat = seeded_island_run(Some(0.5));

    assert_eq!(flat.best_fitness.to_bits(), baseline.best_fitness.to_bits());
    assert_eq!(
        flat.best_individual.genome.genes(),
        baseline.best_individual.genome.genes()
    );
    assert_eq!(flat.fitness_history, baseline.fitness_history);
}

#[test]
fn test_island_mutation_rate_end_changes_the_search() {
    let baseline = seeded_island_run(None);
    let decayed = seeded_island_run(Some(0.0));

    assert_ne!(
        decayed.best_fitness.to_bits(),
        baseline.best_fitness.to_bits(),
        "decaying the mutation rate should change the seeded search"
    );
}

#[test]
fn test_island_rejects_invalid_mutation_rate_end() {
    for bad_rate in [-0.1, 1.5, f64::NAN] {
        let config = island_config_builder(5)
            .mutation_rate_end(bad_rate)
            .build()
            .unwrap();
        let error = config.validate().unwrap_err().to_string();
        assert!(
            error.contains("mutation_rate_end"),
            "expected the error to name the parameter, got: {error}"
        );
    }
}

#[test]
fn test_island_and_serial_report_the_same_generation_count() {
    for generations in [3, 7] {
        let island_config = island_config_builder(generations).build().unwrap();
        let mut island: IslandModel<RealGenome, _> =
            IslandModel::new(island_config, Sphere).unwrap();
        let island_result = island.run();

        let serial_config = GaConfig::builder()
            .population_size(20)
            .genome_length(3)
            .generations(generations)
            .seed(7)
            .build()
            .unwrap();
        let mut serial = GeneticAlgorithm::new(serial_config, Sphere).unwrap();
        let serial_result = serial.run();

        assert_eq!(island_result.generations, serial_result.generations);
        assert_eq!(island_result.generations, generations);
        assert_eq!(
            island_result.fitness_history.len(),
            serial_result.fitness_history.len()
        );
    }
}

#[test]
fn test_island_run_after_step_does_not_rewind_the_generation_counter() {
    let steps = 3;
    let generations = 4;
    let config = island_config_builder(generations).build().unwrap();
    let mut model: IslandModel<RealGenome, _> = IslandModel::new(config, Sphere).unwrap();

    for _ in 0..steps {
        model.step();
    }
    assert_eq!(model.generation(), steps);

    let result = model.run();

    // `run()` must continue from where `step()` left off, not overwrite the
    // counter with its own loop index.
    assert_eq!(model.generation(), steps + generations);
    assert_eq!(result.generations, steps + generations);
    assert!(
        result.generations >= steps,
        "run() must not rewind below the generations already evolved"
    );
}
