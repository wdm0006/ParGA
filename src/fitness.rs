//! Fitness functions for genetic algorithms.
//!
//! This module provides the [`FitnessFunction`] trait and common benchmark functions.

use crate::genome::Genome;

/// Trait for fitness functions.
///
/// Fitness functions evaluate how good a solution (genome) is.
/// Higher fitness values are considered better.
pub trait FitnessFunction<G: Genome>: Clone {
    /// Evaluates the fitness of a genome.
    ///
    /// Returns a value where higher is better.
    fn evaluate(&self, genome: &G) -> f64;
}

/// Wrapper for function pointers as fitness functions.
#[derive(Clone, Copy)]
pub struct FnFitness<G: Genome, F: Fn(&G) -> f64 + Clone>(pub F, std::marker::PhantomData<G>);

impl<G: Genome, F: Fn(&G) -> f64 + Clone> FnFitness<G, F> {
    /// Creates a new function-based fitness evaluator.
    pub fn new(f: F) -> Self {
        Self(f, std::marker::PhantomData)
    }
}

impl<G: Genome, F: Fn(&G) -> f64 + Clone> FitnessFunction<G> for FnFitness<G, F> {
    fn evaluate(&self, genome: &G) -> f64 {
        (self.0)(genome)
    }
}

/// Creates a fitness function from a closure.
pub fn fitness_fn<G: Genome, F: Fn(&G) -> f64 + Clone>(f: F) -> FnFitness<G, F> {
    FnFitness::new(f)
}

/// Benchmark fitness functions for testing and comparison.
pub mod benchmarks {
    use super::{FitnessFunction, Genome};
    use crate::genome::RealGenome;
    use std::f64::consts::PI;

    /// Sphere function (De Jong's function 1).
    ///
    /// Global minimum: f(0, 0, ..., 0) = 0
    /// Search domain: [-5.12, 5.12]^n
    #[derive(Debug, Clone, Copy, Default)]
    pub struct Sphere;

    impl FitnessFunction<RealGenome> for Sphere {
        fn evaluate(&self, genome: &RealGenome) -> f64 {
            // Negate because we maximize fitness
            -genome.genes().iter().map(|x| x * x).sum::<f64>()
        }
    }

    /// Rastrigin function.
    ///
    /// A highly multimodal function with many local minima.
    /// Global minimum: f(0, 0, ..., 0) = 0
    /// Search domain: [-5.12, 5.12]^n
    #[derive(Debug, Clone, Copy, Default)]
    pub struct Rastrigin;

    impl FitnessFunction<RealGenome> for Rastrigin {
        fn evaluate(&self, genome: &RealGenome) -> f64 {
            let n = genome.len() as f64;
            let sum: f64 = genome
                .genes()
                .iter()
                .map(|&x| x * x - 10.0 * (2.0 * PI * x).cos())
                .sum();
            -(10.0 * n + sum)
        }
    }

    /// Rosenbrock function (banana function).
    ///
    /// A classic benchmark with a narrow, curved valley.
    /// Global minimum: f(1, 1, ..., 1) = 0
    /// Search domain: [-5, 10]^n or [-2.048, 2.048]^n
    #[derive(Debug, Clone, Copy, Default)]
    pub struct Rosenbrock;

    impl FitnessFunction<RealGenome> for Rosenbrock {
        fn evaluate(&self, genome: &RealGenome) -> f64 {
            let genes = genome.genes();
            if genes.len() < 2 {
                return 0.0;
            }

            let sum: f64 = genes
                .windows(2)
                .map(|w| {
                    let x = w[0];
                    let y = w[1];
                    100.0 * (y - x * x).powi(2) + (1.0 - x).powi(2)
                })
                .sum();

            -sum
        }
    }

    /// Ackley function.
    ///
    /// A function with many local minima and a global minimum at the origin.
    /// Global minimum: f(0, 0, ..., 0) = 0
    /// Search domain: [-32.768, 32.768]^n
    #[derive(Debug, Clone, Copy, Default)]
    pub struct Ackley;

    impl FitnessFunction<RealGenome> for Ackley {
        fn evaluate(&self, genome: &RealGenome) -> f64 {
            let genes = genome.genes();
            let n = genes.len() as f64;

            if n == 0.0 {
                return 0.0;
            }

            let sum_sq: f64 = genes.iter().map(|x| x * x).sum();
            let sum_cos: f64 = genes.iter().map(|x| (2.0 * PI * x).cos()).sum();

            let term1 = -20.0 * (-0.2 * (sum_sq / n).sqrt()).exp();
            let term2 = -(sum_cos / n).exp();

            -(term1 + term2 + 20.0 + std::f64::consts::E)
        }
    }

    /// Griewank function.
    ///
    /// A function with many widespread local minima.
    /// Global minimum: f(0, 0, ..., 0) = 0
    /// Search domain: [-600, 600]^n
    #[derive(Debug, Clone, Copy, Default)]
    pub struct Griewank;

    impl FitnessFunction<RealGenome> for Griewank {
        fn evaluate(&self, genome: &RealGenome) -> f64 {
            let genes = genome.genes();

            let sum: f64 = genes.iter().map(|x| x * x / 4000.0).sum();

            let prod: f64 = genes
                .iter()
                .enumerate()
                .map(|(i, &x)| (x / ((i + 1) as f64).sqrt()).cos())
                .product();

            -(sum - prod + 1.0)
        }
    }

    /// Schwefel function.
    ///
    /// A deceptive function where the global minimum is far from other local minima.
    /// Global minimum: f(420.9687, ..., 420.9687) ≈ 0
    /// Search domain: [-500, 500]^n
    #[derive(Debug, Clone, Copy, Default)]
    pub struct Schwefel;

    impl FitnessFunction<RealGenome> for Schwefel {
        fn evaluate(&self, genome: &RealGenome) -> f64 {
            let genes = genome.genes();
            let n = genes.len() as f64;

            let sum: f64 = genes.iter().map(|&x| x * x.abs().sqrt().sin()).sum();

            -(418.9829 * n - sum)
        }
    }

    /// Michalewicz function.
    ///
    /// A multimodal function with steep ridges and valleys.
    /// Search domain: [0, π]^n
    #[derive(Debug, Clone, Copy)]
    pub struct Michalewicz {
        /// Steepness parameter (default: 10).
        pub m: f64,
    }

    impl Default for Michalewicz {
        fn default() -> Self {
            Self { m: 10.0 }
        }
    }

    impl FitnessFunction<RealGenome> for Michalewicz {
        fn evaluate(&self, genome: &RealGenome) -> f64 {
            let genes = genome.genes();

            let sum: f64 = genes
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    x.sin() * ((i + 1) as f64 * x * x / PI).sin().powi(2 * self.m as i32)
                })
                .sum();

            sum // Already negative in standard form, but we want to maximize
        }
    }

    /// Booth function.
    ///
    /// A simple 2D function.
    /// Global minimum: f(1, 3) = 0
    /// Search domain: [-10, 10]^2
    #[derive(Debug, Clone, Copy, Default)]
    pub struct Booth;

    impl FitnessFunction<RealGenome> for Booth {
        fn evaluate(&self, genome: &RealGenome) -> f64 {
            let genes = genome.genes();
            if genes.len() < 2 {
                return f64::NEG_INFINITY;
            }

            let x = genes[0];
            let y = genes[1];

            let term1 = (x + 2.0 * y - 7.0).powi(2);
            let term2 = (2.0 * x + y - 5.0).powi(2);

            -(term1 + term2)
        }
    }

    /// Matyas function.
    ///
    /// A simple 2D function.
    /// Global minimum: f(0, 0) = 0
    /// Search domain: [-10, 10]^2
    #[derive(Debug, Clone, Copy, Default)]
    pub struct Matyas;

    impl FitnessFunction<RealGenome> for Matyas {
        fn evaluate(&self, genome: &RealGenome) -> f64 {
            let genes = genome.genes();
            if genes.len() < 2 {
                return f64::NEG_INFINITY;
            }

            let x = genes[0];
            let y = genes[1];

            -(0.26 * (x * x + y * y) - 0.48 * x * y)
        }
    }

    /// Himmelblau function.
    ///
    /// A 2D function with four identical local minima.
    /// Global minima at: (3, 2), (-2.805, 3.131), (-3.779, -3.283), (3.584, -1.848)
    /// All with f = 0
    /// Search domain: [-5, 5]^2
    #[derive(Debug, Clone, Copy, Default)]
    pub struct Himmelblau;

    impl FitnessFunction<RealGenome> for Himmelblau {
        fn evaluate(&self, genome: &RealGenome) -> f64 {
            let genes = genome.genes();
            if genes.len() < 2 {
                return f64::NEG_INFINITY;
            }

            let x = genes[0];
            let y = genes[1];

            let term1 = (x * x + y - 11.0).powi(2);
            let term2 = (x + y * y - 7.0).powi(2);

            -(term1 + term2)
        }
    }

    /// Easom function.
    ///
    /// A 2D function with a very small region containing the global minimum.
    /// Global minimum: f(π, π) = -1
    /// Search domain: [-100, 100]^2
    #[derive(Debug, Clone, Copy, Default)]
    pub struct Easom;

    impl FitnessFunction<RealGenome> for Easom {
        fn evaluate(&self, genome: &RealGenome) -> f64 {
            let genes = genome.genes();
            if genes.len() < 2 {
                return f64::NEG_INFINITY;
            }

            let x = genes[0];
            let y = genes[1];

            -(-x.cos() * y.cos() * (-(x - PI).powi(2) - (y - PI).powi(2)).exp())
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_sphere_at_origin() {
            let genome = RealGenome::new(vec![0.0, 0.0, 0.0]);
            let fitness = Sphere.evaluate(&genome);
            assert!((fitness - 0.0).abs() < 1e-10);
        }

        #[test]
        fn test_rastrigin_at_origin() {
            let genome = RealGenome::new(vec![0.0, 0.0]);
            let fitness = Rastrigin.evaluate(&genome);
            assert!((fitness - 0.0).abs() < 1e-10);
        }

        #[test]
        fn test_rosenbrock_at_optimum() {
            let genome = RealGenome::new(vec![1.0, 1.0, 1.0]);
            let fitness = Rosenbrock.evaluate(&genome);
            assert!((fitness - 0.0).abs() < 1e-10);
        }

        #[test]
        fn test_ackley_at_origin() {
            let genome = RealGenome::new(vec![0.0, 0.0]);
            let fitness = Ackley.evaluate(&genome);
            assert!((fitness - 0.0).abs() < 1e-10);
        }

        #[test]
        fn test_booth_at_optimum() {
            let genome = RealGenome::new(vec![1.0, 3.0]);
            let fitness = Booth.evaluate(&genome);
            assert!((fitness - 0.0).abs() < 1e-10);
        }
    }
}
