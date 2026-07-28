//! Integration tests for the parga library.

use parga::prelude::*;
use parga::{
    fitness::fitness_fn,
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
