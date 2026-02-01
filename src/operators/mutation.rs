//! Mutation operators for genetic algorithms.
//!
//! Mutation introduces random variation to maintain genetic diversity.

use crate::genome::{BinaryGenome, Genome, PermutationGenome, RealGenome};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

/// Trait for mutation operators.
pub trait Mutation<G: Genome> {
    /// Mutates the genome in place.
    fn mutate<R: Rng>(&self, genome: &mut G, rate: f64, lower: &[f64], upper: &[f64], rng: &mut R);
}

/// Available mutation methods for real-valued genomes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RealMutation {
    /// Gaussian mutation with standard deviation as fraction of range.
    Gaussian(f64),

    /// Uniform mutation within bounds.
    Uniform,

    /// Polynomial mutation with distribution index.
    Polynomial(f64),

    /// Non-uniform mutation (decreases over time).
    NonUniform {
        generation: usize,
        max_generations: usize,
    },

    /// Boundary mutation (set to bound).
    Boundary,
}

/// Available mutation methods for binary genomes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinaryMutation {
    /// Flip bits with given probability.
    BitFlip,

    /// Flip a single random bit.
    SingleBitFlip,
}

/// Available mutation methods for permutation genomes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PermutationMutation {
    /// Swap two random positions.
    Swap,

    /// Insert element at random position.
    Insert,

    /// Reverse a random segment.
    Inversion,

    /// Scramble a random segment.
    Scramble,
}

/// Generic mutation operator that handles different genome types.
#[derive(Debug, Clone, Copy)]
pub enum MutationOperator<G: Genome> {
    /// Mutation for real-valued genomes.
    Real(RealMutation),

    /// Mutation for binary genomes.
    Binary(BinaryMutation),

    /// Mutation for permutation genomes.
    Permutation(PermutationMutation),

    /// Phantom data to satisfy the type system.
    #[doc(hidden)]
    _Phantom(std::marker::PhantomData<G>),
}

impl<G: Genome> Default for MutationOperator<G> {
    fn default() -> Self {
        Self::Real(RealMutation::Gaussian(0.1))
    }
}

impl Mutation<RealGenome> for MutationOperator<RealGenome> {
    fn mutate<R: Rng>(
        &self,
        genome: &mut RealGenome,
        rate: f64,
        lower: &[f64],
        upper: &[f64],
        rng: &mut R,
    ) {
        match self {
            Self::Real(op) => op.mutate(genome, rate, lower, upper, rng),
            _ => RealMutation::Gaussian(0.1).mutate(genome, rate, lower, upper, rng),
        }
    }
}

impl Mutation<BinaryGenome> for MutationOperator<BinaryGenome> {
    fn mutate<R: Rng>(
        &self,
        genome: &mut BinaryGenome,
        rate: f64,
        _lower: &[f64],
        _upper: &[f64],
        rng: &mut R,
    ) {
        match self {
            Self::Binary(op) => op.mutate(genome, rate, rng),
            _ => BinaryMutation::BitFlip.mutate(genome, rate, rng),
        }
    }
}

impl Mutation<PermutationGenome> for MutationOperator<PermutationGenome> {
    fn mutate<R: Rng>(
        &self,
        genome: &mut PermutationGenome,
        rate: f64,
        _lower: &[f64],
        _upper: &[f64],
        rng: &mut R,
    ) {
        match self {
            Self::Permutation(op) => op.mutate(genome, rate, rng),
            _ => PermutationMutation::Swap.mutate(genome, rate, rng),
        }
    }
}

impl Mutation<RealGenome> for RealMutation {
    fn mutate<R: Rng>(
        &self,
        genome: &mut RealGenome,
        rate: f64,
        lower: &[f64],
        upper: &[f64],
        rng: &mut R,
    ) {
        let genes = genome.genes_mut();

        match self {
            Self::Gaussian(sigma_factor) => {
                for (i, gene) in genes.iter_mut().enumerate() {
                    if rng.gen::<f64>() < rate {
                        let lo = lower.get(i).copied().unwrap_or(-10.0);
                        let hi = upper.get(i).copied().unwrap_or(10.0);
                        let range = hi - lo;
                        let sigma = range * sigma_factor;

                        let normal =
                            Normal::new(0.0, sigma).unwrap_or(Normal::new(0.0, 1.0).unwrap());
                        let delta = normal.sample(rng);
                        *gene = (*gene + delta).clamp(lo, hi);
                    }
                }
            }

            Self::Uniform => {
                for (i, gene) in genes.iter_mut().enumerate() {
                    if rng.gen::<f64>() < rate {
                        let lo = lower.get(i).copied().unwrap_or(-10.0);
                        let hi = upper.get(i).copied().unwrap_or(10.0);
                        *gene = rng.gen_range(lo..=hi);
                    }
                }
            }

            Self::Polynomial(eta) => {
                for (i, gene) in genes.iter_mut().enumerate() {
                    if rng.gen::<f64>() < rate {
                        let lo = lower.get(i).copied().unwrap_or(-10.0);
                        let hi = upper.get(i).copied().unwrap_or(10.0);
                        let delta = (*gene - lo) / (hi - lo);

                        let u: f64 = rng.gen();
                        let delta_q = if u < 0.5 {
                            let xy = 1.0 - delta;
                            let val = 2.0 * u + (1.0 - 2.0 * u) * xy.powf(eta + 1.0);
                            val.powf(1.0 / (eta + 1.0)) - 1.0
                        } else {
                            let xy = 1.0 - delta;
                            1.0 - (2.0 * (1.0 - u) + 2.0 * (u - 0.5) * xy.powf(eta + 1.0))
                                .powf(1.0 / (eta + 1.0))
                        };

                        *gene = (*gene + delta_q * (hi - lo)).clamp(lo, hi);
                    }
                }
            }

            Self::NonUniform {
                generation,
                max_generations,
            } => {
                let t = *generation as f64 / *max_generations as f64;
                let b = 5.0; // System parameter

                for (i, gene) in genes.iter_mut().enumerate() {
                    if rng.gen::<f64>() < rate {
                        let lo = lower.get(i).copied().unwrap_or(-10.0);
                        let hi = upper.get(i).copied().unwrap_or(10.0);

                        let tau: f64 = rng.gen();
                        let r: f64 = rng.gen();
                        let delta = 1.0 - r.powf((1.0 - t).powf(b));

                        *gene = if tau < 0.5 {
                            (*gene + (hi - *gene) * delta).clamp(lo, hi)
                        } else {
                            (*gene - (*gene - lo) * delta).clamp(lo, hi)
                        };
                    }
                }
            }

            Self::Boundary => {
                for (i, gene) in genes.iter_mut().enumerate() {
                    if rng.gen::<f64>() < rate {
                        let lo = lower.get(i).copied().unwrap_or(-10.0);
                        let hi = upper.get(i).copied().unwrap_or(10.0);
                        *gene = if rng.gen() { lo } else { hi };
                    }
                }
            }
        }
    }
}

impl BinaryMutation {
    /// Mutates a binary genome.
    pub fn mutate<R: Rng>(&self, genome: &mut BinaryGenome, rate: f64, rng: &mut R) {
        match self {
            Self::BitFlip => {
                for i in 0..genome.len() {
                    if rng.gen::<f64>() < rate {
                        genome.flip(i);
                    }
                }
            }

            Self::SingleBitFlip => {
                if rng.gen::<f64>() < rate && !genome.is_empty() {
                    let idx = rng.gen_range(0..genome.len());
                    genome.flip(idx);
                }
            }
        }
    }
}

impl PermutationMutation {
    /// Mutates a permutation genome.
    pub fn mutate<R: Rng>(&self, genome: &mut PermutationGenome, rate: f64, rng: &mut R) {
        if rng.gen::<f64>() >= rate || genome.len() < 2 {
            return;
        }

        let len = genome.len();

        match self {
            Self::Swap => {
                let i = rng.gen_range(0..len);
                let j = rng.gen_range(0..len);
                genome.swap(i, j);
            }

            Self::Insert => {
                let from = rng.gen_range(0..len);
                let to = rng.gen_range(0..len);

                if from != to {
                    let val = genome[from];
                    let order = genome.order_mut();

                    if from < to {
                        for i in from..to {
                            order[i] = order[i + 1];
                        }
                    } else {
                        for i in (to + 1..=from).rev() {
                            order[i] = order[i - 1];
                        }
                    }
                    order[to] = val;
                }
            }

            Self::Inversion => {
                let mut i = rng.gen_range(0..len);
                let mut j = rng.gen_range(0..len);
                if i > j {
                    std::mem::swap(&mut i, &mut j);
                }
                genome.reverse_segment(i, j + 1);
            }

            Self::Scramble => {
                use rand::seq::SliceRandom;

                let mut i = rng.gen_range(0..len);
                let mut j = rng.gen_range(0..len);
                if i > j {
                    std::mem::swap(&mut i, &mut j);
                }

                let order = genome.order_mut();
                order[i..=j].shuffle(rng);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_mutation() {
        let mut genome = RealGenome::new(vec![0.0, 0.0, 0.0]);
        let lower = vec![-10.0, -10.0, -10.0];
        let upper = vec![10.0, 10.0, 10.0];
        let mut rng = crate::rng::create_rng(Some(42));

        RealMutation::Gaussian(0.1).mutate(&mut genome, 1.0, &lower, &upper, &mut rng);

        // At least some genes should have changed
        let changed = genome.genes().iter().any(|&g| g != 0.0);
        assert!(changed);

        // All genes should be within bounds
        for (i, &g) in genome.genes().iter().enumerate() {
            assert!(g >= lower[i] && g <= upper[i]);
        }
    }

    #[test]
    fn test_bit_flip_mutation() {
        let mut genome = BinaryGenome::from_bools(&[false, false, false, false]);
        let mut rng = crate::rng::create_rng(Some(42));

        BinaryMutation::BitFlip.mutate(&mut genome, 1.0, &mut rng);

        // All bits should have flipped
        assert!(genome.bits().iter().all(|b| *b));
    }

    #[test]
    fn test_swap_mutation() {
        let mut genome = PermutationGenome::new(vec![0, 1, 2, 3, 4]);
        let original = genome.order().to_vec();
        let mut rng = crate::rng::create_rng(Some(42));

        PermutationMutation::Swap.mutate(&mut genome, 1.0, &mut rng);

        // Should still be a valid permutation
        let mut sorted: Vec<_> = genome.order().to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);

        // Order might have changed
        let _ = original; // Original stored for reference
    }

    #[test]
    fn test_inversion_mutation() {
        let mut genome = PermutationGenome::new(vec![0, 1, 2, 3, 4]);
        let mut rng = crate::rng::create_rng(Some(42));

        PermutationMutation::Inversion.mutate(&mut genome, 1.0, &mut rng);

        // Should still be a valid permutation
        let mut sorted: Vec<_> = genome.order().to_vec();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }
}
