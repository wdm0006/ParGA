//! Island model genetic algorithm example.
//!
//! This example demonstrates how to use the island model for parallel
//! genetic algorithm optimization, which is particularly effective for
//! multimodal functions like Rastrigin.

use parga::prelude::*;

fn main() {
    println!("ParGA Example: Island Model\n");

    // Configure the island model
    let config = IslandConfig::builder()
        .num_islands(4)
        .island_population(50)
        .genome_length(10)
        .generations(100)
        .migration_interval(20)
        .migration_count(5)
        .topology(MigrationTopology::Ring)
        .mutation_rate(0.05)
        .crossover_rate(0.9)
        .elitism(2)
        .tournament_size(5)
        .lower_bounds(vec![-5.12; 10])
        .upper_bounds(vec![5.12; 10])
        .seed(42)
        .build()
        .expect("Failed to build config");

    // Use the Rastrigin function (highly multimodal)
    let fitness = Rastrigin;

    println!("Optimizing 10-dimensional Rastrigin function");
    println!("Using {} islands with {} individuals each", 4, 50);
    println!("Migration every {} generations\n", 20);

    // Create and run the island model
    let mut model: IslandModel<RealGenome, _> = IslandModel::new(config, fitness);
    let result = model.run();

    // Print results
    println!("Optimization complete!");
    println!("Best fitness: {:.6}", result.best_fitness);
    println!("(Optimal is 0.0)");
    println!("Generations: {}", result.generations);
    println!("Converged: {}", result.converged);
    println!(
        "\nBest fitness per island: {:?}",
        result
            .island_best_fitness
            .iter()
            .map(|f| format!("{:.4}", f))
            .collect::<Vec<_>>()
    );
    println!(
        "\nBest solution: {:?}",
        result
            .best_individual
            .genome
            .genes()
            .iter()
            .map(|x| format!("{:.4}", x))
            .collect::<Vec<_>>()
    );

    // Try different topologies
    println!("\n--- Comparing Migration Topologies ---\n");

    let topologies = [
        ("Ring", MigrationTopology::Ring),
        ("Star", MigrationTopology::Star),
        ("Ladder", MigrationTopology::Ladder),
        ("Fully Connected", MigrationTopology::FullyConnected),
    ];

    for (name, topology) in topologies {
        let config = IslandConfig::builder()
            .num_islands(4)
            .island_population(30)
            .genome_length(10)
            .generations(50)
            .migration_interval(10)
            .migration_count(3)
            .topology(topology)
            .lower_bounds(vec![-5.12; 10])
            .upper_bounds(vec![5.12; 10])
            .seed(42)
            .build()
            .unwrap();

        let mut model: IslandModel<RealGenome, _> = IslandModel::new(config, Rastrigin);
        let result = model.run();

        println!("{:20} Best fitness: {:.6}", name, result.best_fitness);
    }
}
