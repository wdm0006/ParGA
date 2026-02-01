//! Basic genetic algorithm example.
//!
//! This example demonstrates how to use the parga library to optimize
//! the Sphere function using a simple genetic algorithm.

use parga::prelude::*;

fn main() {
    println!("ParGA Example: Basic Genetic Algorithm\n");

    // Configure the genetic algorithm
    let config = GaConfig::builder()
        .population_size(100)
        .genome_length(10)
        .generations(100)
        .mutation_rate(0.02)
        .crossover_rate(0.8)
        .elitism(2)
        .tournament_size(3)
        .lower_bounds(vec![-5.0; 10])
        .upper_bounds(vec![5.0; 10])
        .seed(42) // For reproducibility
        .build()
        .expect("Failed to build config");

    // Use the built-in Sphere function
    let fitness = Sphere;

    // Create and run the genetic algorithm
    let mut ga: GeneticAlgorithm<RealGenome, _> = GeneticAlgorithm::new(config, fitness);
    let result = ga.run();

    // Print results
    println!("Optimization complete!");
    println!("Best fitness: {:.6}", result.best_fitness);
    println!("Generations: {}", result.generations);
    println!("Converged: {}", result.converged);
    println!(
        "Best solution: {:?}",
        result
            .best_individual
            .genome
            .genes()
            .iter()
            .map(|x| format!("{:.4}", x))
            .collect::<Vec<_>>()
    );

    // The optimal solution for Sphere is (0, 0, ..., 0) with fitness 0
    let distance_from_optimum: f64 = result
        .best_individual
        .genome
        .genes()
        .iter()
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt();

    println!("Distance from optimum: {:.6}", distance_from_optimum);
}
