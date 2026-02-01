//! Genome types for different optimization problems.
//!
//! This module provides three main genome representations:
//! - [`RealGenome`]: Continuous real-valued genes (most common)
//! - [`BinaryGenome`]: Binary bit strings
//! - [`PermutationGenome`]: Ordered permutations (for TSP-like problems)

use bitvec::prelude::*;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// Trait for genome types that can be evolved.
pub trait Genome: Clone + Debug + Send + Sync + Default + 'static {
    /// The type of a single gene.
    type Gene: Clone + Debug;

    /// Returns the length of the genome.
    fn len(&self) -> usize;

    /// Returns true if the genome is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Creates a random genome within the given bounds.
    fn random<R: Rng>(rng: &mut R, length: usize, lower: &[f64], upper: &[f64]) -> Self;

    /// Returns the genes as a vector of f64 for fitness evaluation.
    fn as_f64_vec(&self) -> Vec<f64>;

    /// Creates a genome from a vector of f64 values.
    fn from_f64_vec(values: Vec<f64>) -> Self;
}

/// Real-valued genome for continuous optimization.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct RealGenome {
    genes: Vec<f64>,
}

impl RealGenome {
    /// Creates a new real genome from a vector of genes.
    pub fn new(genes: Vec<f64>) -> Self {
        Self { genes }
    }

    /// Returns a reference to the genes.
    pub fn genes(&self) -> &[f64] {
        &self.genes
    }

    /// Returns a mutable reference to the genes.
    pub fn genes_mut(&mut self) -> &mut [f64] {
        &mut self.genes
    }

    /// Gets a gene at the given index.
    pub fn get(&self, index: usize) -> Option<f64> {
        self.genes.get(index).copied()
    }

    /// Sets a gene at the given index.
    pub fn set(&mut self, index: usize, value: f64) {
        if index < self.genes.len() {
            self.genes[index] = value;
        }
    }

    /// Clamps all genes to the given bounds.
    pub fn clamp(&mut self, lower: &[f64], upper: &[f64]) {
        for (i, gene) in self.genes.iter_mut().enumerate() {
            let lo = lower.get(i).copied().unwrap_or(f64::NEG_INFINITY);
            let hi = upper.get(i).copied().unwrap_or(f64::INFINITY);
            *gene = gene.clamp(lo, hi);
        }
    }
}

impl Genome for RealGenome {
    type Gene = f64;

    fn len(&self) -> usize {
        self.genes.len()
    }

    fn random<R: Rng>(rng: &mut R, length: usize, lower: &[f64], upper: &[f64]) -> Self {
        let genes: Vec<f64> = (0..length)
            .map(|i| {
                let lo = lower.get(i).copied().unwrap_or(-10.0);
                let hi = upper.get(i).copied().unwrap_or(10.0);
                rng.gen_range(lo..=hi)
            })
            .collect();
        Self { genes }
    }

    fn as_f64_vec(&self) -> Vec<f64> {
        self.genes.clone()
    }

    fn from_f64_vec(values: Vec<f64>) -> Self {
        Self::new(values)
    }
}

impl std::ops::Index<usize> for RealGenome {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.genes[index]
    }
}

impl std::ops::IndexMut<usize> for RealGenome {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.genes[index]
    }
}

/// Binary genome for discrete optimization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BinaryGenome {
    #[serde(with = "bitvec_serde")]
    bits: BitVec<u64, Lsb0>,
}

mod bitvec_serde {
    use bitvec::prelude::*;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(bits: &BitVec<u64, Lsb0>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let bytes: Vec<bool> = bits.iter().map(|b| *b).collect();
        bytes.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<BitVec<u64, Lsb0>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let bytes: Vec<bool> = Vec::deserialize(deserializer)?;
        Ok(bytes.into_iter().collect())
    }
}

impl BinaryGenome {
    /// Creates a new binary genome from a bit vector.
    pub fn new(bits: BitVec<u64, Lsb0>) -> Self {
        Self { bits }
    }

    /// Creates a binary genome from a slice of booleans.
    pub fn from_bools(bools: &[bool]) -> Self {
        Self {
            bits: bools.iter().collect(),
        }
    }

    /// Returns a reference to the bits.
    pub fn bits(&self) -> &BitVec<u64, Lsb0> {
        &self.bits
    }

    /// Returns a mutable reference to the bits.
    pub fn bits_mut(&mut self) -> &mut BitVec<u64, Lsb0> {
        &mut self.bits
    }

    /// Gets a bit at the given index.
    pub fn get(&self, index: usize) -> Option<bool> {
        self.bits.get(index).map(|b| *b)
    }

    /// Sets a bit at the given index.
    pub fn set(&mut self, index: usize, value: bool) {
        if index < self.bits.len() {
            self.bits.set(index, value);
        }
    }

    /// Flips a bit at the given index.
    pub fn flip(&mut self, index: usize) {
        if index < self.bits.len() {
            let current = self.bits[index];
            self.bits.set(index, !current);
        }
    }

    /// Counts the number of set bits.
    pub fn count_ones(&self) -> usize {
        self.bits.count_ones()
    }

    /// Decodes the binary genome to a real value in [0, 1].
    pub fn decode_normalized(&self) -> f64 {
        if self.bits.is_empty() {
            return 0.0;
        }
        let max_val = (1u128 << self.bits.len().min(64)) - 1;
        let val: u64 = self.bits.iter().enumerate().fold(0u64, |acc, (i, bit)| {
            if *bit && i < 64 {
                acc | (1u64 << i)
            } else {
                acc
            }
        });
        val as f64 / max_val as f64
    }

    /// Decodes the binary genome to a real value in [lower, upper].
    pub fn decode(&self, lower: f64, upper: f64) -> f64 {
        lower + self.decode_normalized() * (upper - lower)
    }
}

impl Default for BinaryGenome {
    fn default() -> Self {
        Self {
            bits: BitVec::new(),
        }
    }
}

impl Genome for BinaryGenome {
    type Gene = bool;

    fn len(&self) -> usize {
        self.bits.len()
    }

    fn random<R: Rng>(rng: &mut R, length: usize, _lower: &[f64], _upper: &[f64]) -> Self {
        let bits: BitVec<u64, Lsb0> = (0..length).map(|_| rng.gen::<bool>()).collect();
        Self { bits }
    }

    fn as_f64_vec(&self) -> Vec<f64> {
        self.bits
            .iter()
            .map(|b| if *b { 1.0 } else { 0.0 })
            .collect()
    }

    fn from_f64_vec(values: Vec<f64>) -> Self {
        Self::from_bools(&values.iter().map(|v| *v >= 0.5).collect::<Vec<_>>())
    }
}

/// Permutation genome for ordering problems (TSP, scheduling).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct PermutationGenome {
    order: Vec<usize>,
}

impl PermutationGenome {
    /// Creates a new permutation genome from a vector of indices.
    ///
    /// # Panics
    ///
    /// Panics if the indices don't form a valid permutation.
    pub fn new(order: Vec<usize>) -> Self {
        debug_assert!(Self::is_valid_permutation(&order));
        Self { order }
    }

    /// Creates a sequential permutation [0, 1, 2, ..., n-1].
    pub fn sequential(n: usize) -> Self {
        Self {
            order: (0..n).collect(),
        }
    }

    /// Returns a reference to the order.
    pub fn order(&self) -> &[usize] {
        &self.order
    }

    /// Returns a mutable reference to the order.
    pub fn order_mut(&mut self) -> &mut [usize] {
        &mut self.order
    }

    /// Gets the element at the given position.
    pub fn get(&self, index: usize) -> Option<usize> {
        self.order.get(index).copied()
    }

    /// Swaps two positions in the permutation.
    pub fn swap(&mut self, i: usize, j: usize) {
        if i < self.order.len() && j < self.order.len() {
            self.order.swap(i, j);
        }
    }

    /// Reverses a segment of the permutation.
    pub fn reverse_segment(&mut self, start: usize, end: usize) {
        if start < end && end <= self.order.len() {
            self.order[start..end].reverse();
        }
    }

    /// Checks if a vector is a valid permutation.
    fn is_valid_permutation(order: &[usize]) -> bool {
        let n = order.len();
        let mut seen = vec![false; n];
        for &i in order {
            if i >= n || seen[i] {
                return false;
            }
            seen[i] = true;
        }
        true
    }

    /// Repairs the permutation to ensure validity after crossover.
    #[allow(clippy::needless_range_loop)]
    pub fn repair(&mut self) {
        let n = self.order.len();
        let mut seen = vec![false; n];
        let mut missing: Vec<usize> = Vec::new();

        // Find duplicates and missing values
        for i in 0..n {
            if self.order[i] >= n {
                self.order[i] = 0; // Reset invalid indices
            }
            if seen[self.order[i]] {
                self.order[i] = n; // Mark for replacement
            } else {
                seen[self.order[i]] = true;
            }
        }

        // Collect missing values
        for i in 0..n {
            if !seen[i] {
                missing.push(i);
            }
        }

        // Replace duplicates with missing values
        let mut missing_idx = 0;
        for i in 0..n {
            if self.order[i] == n {
                self.order[i] = missing[missing_idx];
                missing_idx += 1;
            }
        }
    }
}

impl Genome for PermutationGenome {
    type Gene = usize;

    fn len(&self) -> usize {
        self.order.len()
    }

    fn random<R: Rng>(rng: &mut R, length: usize, _lower: &[f64], _upper: &[f64]) -> Self {
        use rand::seq::SliceRandom;
        let mut order: Vec<usize> = (0..length).collect();
        order.shuffle(rng);
        Self { order }
    }

    fn as_f64_vec(&self) -> Vec<f64> {
        self.order.iter().map(|&i| i as f64).collect()
    }

    fn from_f64_vec(values: Vec<f64>) -> Self {
        Self::new(values.iter().map(|v| *v as usize).collect())
    }
}

impl std::ops::Index<usize> for PermutationGenome {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.order[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_real_genome() {
        let genome = RealGenome::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(genome.len(), 3);
        assert_eq!(genome[0], 1.0);
        assert_eq!(genome.as_f64_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_real_genome_random() {
        let mut rng = crate::rng::create_rng(Some(42));
        let lower = vec![-1.0, -1.0];
        let upper = vec![1.0, 1.0];
        let genome = RealGenome::random(&mut rng, 2, &lower, &upper);
        assert_eq!(genome.len(), 2);
        assert!(genome[0] >= -1.0 && genome[0] <= 1.0);
        assert!(genome[1] >= -1.0 && genome[1] <= 1.0);
    }

    #[test]
    fn test_binary_genome() {
        let genome = BinaryGenome::from_bools(&[true, false, true, true]);
        assert_eq!(genome.len(), 4);
        assert_eq!(genome.get(0), Some(true));
        assert_eq!(genome.get(1), Some(false));
        assert_eq!(genome.count_ones(), 3);
    }

    #[test]
    fn test_binary_decode() {
        let genome = BinaryGenome::from_bools(&[true, true, true, true]);
        let decoded = genome.decode(0.0, 10.0);
        assert!((decoded - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_permutation_genome() {
        let genome = PermutationGenome::new(vec![2, 0, 1, 3]);
        assert_eq!(genome.len(), 4);
        assert_eq!(genome[0], 2);
        assert_eq!(genome.order(), &[2, 0, 1, 3]);
    }

    #[test]
    fn test_permutation_repair() {
        let mut genome = PermutationGenome::new(vec![0, 1, 2, 3]);
        genome.order_mut()[1] = 0; // Create duplicate
        genome.repair();
        assert!(PermutationGenome::is_valid_permutation(genome.order()));
    }
}
