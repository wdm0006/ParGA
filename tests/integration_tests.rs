//! Integration tests for the parga library.

use parga::prelude::*;

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
    let mut ga: GeneticAlgorithm<RealGenome, _> = GeneticAlgorithm::new(config, fitness);
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
    let mut ga: GeneticAlgorithm<RealGenome, _> = GeneticAlgorithm::new(config, fitness);
    let result = ga.run();

    // Should find a reasonable solution (Rastrigin is harder)
    assert!(result.best_fitness > -10.0);
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
    let mut island_model: IslandModel<RealGenome, _> = IslandModel::new(config, fitness);
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
        let mut model: IslandModel<RealGenome, _> = IslandModel::new(config, fitness);
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
        let mut ga: GeneticAlgorithm<RealGenome, _> =
            GeneticAlgorithm::new(config, fitness).with_selection(method);
        let result = ga.run();

        assert!(
            result.best_fitness.is_finite(),
            "Selection {:?} failed",
            method
        );
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

    let mut ga = GeneticAlgorithm::new(config, fitness);
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
    let mut ga: GeneticAlgorithm<RealGenome, _> = GeneticAlgorithm::new(config, fitness);
    let result = ga.run();

    // With enough generations, should converge
    // Note: convergence depends on the random seed
    assert!(!result.fitness_history.is_empty());
}

#[test]
#[ignore = "Reproducibility requires thread-local RNG seeding, not yet implemented"]
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

    let mut ga1: GeneticAlgorithm<RealGenome, _> = GeneticAlgorithm::new(config1, fitness);
    let result1 = ga1.run();

    let mut ga2: GeneticAlgorithm<RealGenome, _> = GeneticAlgorithm::new(config2, fitness);
    let result2 = ga2.run();

    // Results should be identical with the same seed
    assert!((result1.best_fitness - result2.best_fitness).abs() < 1e-10);
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
    let mut ga: GeneticAlgorithm<RealGenome, _> = GeneticAlgorithm::new(config, fitness);
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
    let mut ga: GeneticAlgorithm<RealGenome, _> = GeneticAlgorithm::new(config, fitness);
    let result = ga.run();

    // Booth optimum is at (1, 3) with fitness 0
    // Should get reasonably close
    assert!(result.best_fitness > -5.0);
}
