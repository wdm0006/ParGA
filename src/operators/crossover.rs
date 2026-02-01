//! Crossover operators for genetic algorithms.
//!
//! Crossover combines genetic material from two parents to create offspring.

use crate::genome::{BinaryGenome, Genome, PermutationGenome, RealGenome};
use bitvec::prelude::*;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Trait for crossover operators.
pub trait Crossover<G: Genome> {
    /// Performs crossover between two parents, returning two offspring.
    fn crossover<R: Rng>(&self, parent1: &G, parent2: &G, rng: &mut R) -> (G, G);
}

/// Available crossover methods for real-valued genomes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RealCrossover {
    /// Single-point crossover.
    SinglePoint,

    /// Two-point crossover.
    TwoPoint,

    /// Uniform crossover with given probability.
    Uniform(f64),

    /// Blend crossover (BLX-alpha) for continuous spaces.
    Blend(f64),

    /// Simulated Binary Crossover (SBX) with given distribution index.
    SimulatedBinary(f64),

    /// Arithmetic crossover (weighted average).
    Arithmetic,
}

/// Available crossover methods for binary genomes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum BinaryCrossover {
    /// Single-point crossover.
    SinglePoint,

    /// Two-point crossover.
    TwoPoint,

    /// Uniform crossover with given probability.
    Uniform(f64),
}

/// Available crossover methods for permutation genomes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PermutationCrossover {
    /// Order crossover (OX1).
    Order,

    /// Partially-mapped crossover (PMX).
    PartiallyMapped,

    /// Cycle crossover (CX).
    Cycle,

    /// Edge recombination crossover.
    EdgeRecombination,
}

/// Generic crossover operator that handles different genome types.
#[derive(Debug, Clone, Copy)]
pub enum CrossoverOperator<G: Genome> {
    /// Crossover for real-valued genomes.
    Real(RealCrossover),

    /// Crossover for binary genomes.
    Binary(BinaryCrossover),

    /// Crossover for permutation genomes.
    Permutation(PermutationCrossover),

    /// Phantom data to satisfy the type system.
    #[doc(hidden)]
    _Phantom(std::marker::PhantomData<G>),
}

impl<G: Genome> Default for CrossoverOperator<G> {
    fn default() -> Self {
        Self::Real(RealCrossover::TwoPoint)
    }
}

impl Crossover<RealGenome> for CrossoverOperator<RealGenome> {
    fn crossover<R: Rng>(
        &self,
        parent1: &RealGenome,
        parent2: &RealGenome,
        rng: &mut R,
    ) -> (RealGenome, RealGenome) {
        match self {
            Self::Real(op) => op.crossover(parent1, parent2, rng),
            _ => RealCrossover::TwoPoint.crossover(parent1, parent2, rng),
        }
    }
}

impl Crossover<BinaryGenome> for CrossoverOperator<BinaryGenome> {
    fn crossover<R: Rng>(
        &self,
        parent1: &BinaryGenome,
        parent2: &BinaryGenome,
        rng: &mut R,
    ) -> (BinaryGenome, BinaryGenome) {
        match self {
            Self::Binary(op) => op.crossover(parent1, parent2, rng),
            _ => BinaryCrossover::TwoPoint.crossover(parent1, parent2, rng),
        }
    }
}

impl Crossover<PermutationGenome> for CrossoverOperator<PermutationGenome> {
    fn crossover<R: Rng>(
        &self,
        parent1: &PermutationGenome,
        parent2: &PermutationGenome,
        rng: &mut R,
    ) -> (PermutationGenome, PermutationGenome) {
        match self {
            Self::Permutation(op) => op.crossover(parent1, parent2, rng),
            _ => PermutationCrossover::Order.crossover(parent1, parent2, rng),
        }
    }
}

impl Crossover<RealGenome> for RealCrossover {
    fn crossover<R: Rng>(
        &self,
        parent1: &RealGenome,
        parent2: &RealGenome,
        rng: &mut R,
    ) -> (RealGenome, RealGenome) {
        let len = parent1.len().min(parent2.len());
        if len == 0 {
            return (RealGenome::default(), RealGenome::default());
        }

        match self {
            Self::SinglePoint => {
                let point = rng.gen_range(0..=len);
                let mut child1 = Vec::with_capacity(len);
                let mut child2 = Vec::with_capacity(len);

                for i in 0..len {
                    if i < point {
                        child1.push(parent1.genes()[i]);
                        child2.push(parent2.genes()[i]);
                    } else {
                        child1.push(parent2.genes()[i]);
                        child2.push(parent1.genes()[i]);
                    }
                }

                (RealGenome::new(child1), RealGenome::new(child2))
            }

            Self::TwoPoint => {
                let mut p1 = rng.gen_range(0..=len);
                let mut p2 = rng.gen_range(0..=len);
                if p1 > p2 {
                    std::mem::swap(&mut p1, &mut p2);
                }

                let mut child1 = Vec::with_capacity(len);
                let mut child2 = Vec::with_capacity(len);

                for i in 0..len {
                    if i < p1 || i >= p2 {
                        child1.push(parent1.genes()[i]);
                        child2.push(parent2.genes()[i]);
                    } else {
                        child1.push(parent2.genes()[i]);
                        child2.push(parent1.genes()[i]);
                    }
                }

                (RealGenome::new(child1), RealGenome::new(child2))
            }

            Self::Uniform(prob) => {
                let mut child1 = Vec::with_capacity(len);
                let mut child2 = Vec::with_capacity(len);

                for i in 0..len {
                    if rng.gen::<f64>() < *prob {
                        child1.push(parent2.genes()[i]);
                        child2.push(parent1.genes()[i]);
                    } else {
                        child1.push(parent1.genes()[i]);
                        child2.push(parent2.genes()[i]);
                    }
                }

                (RealGenome::new(child1), RealGenome::new(child2))
            }

            Self::Blend(alpha) => {
                let mut child1 = Vec::with_capacity(len);
                let mut child2 = Vec::with_capacity(len);

                for i in 0..len {
                    let p1 = parent1.genes()[i];
                    let p2 = parent2.genes()[i];
                    let diff = (p2 - p1).abs();
                    let min_val = p1.min(p2) - alpha * diff;
                    let max_val = p1.max(p2) + alpha * diff;

                    child1.push(rng.gen_range(min_val..=max_val));
                    child2.push(rng.gen_range(min_val..=max_val));
                }

                (RealGenome::new(child1), RealGenome::new(child2))
            }

            Self::SimulatedBinary(eta) => {
                let mut child1 = Vec::with_capacity(len);
                let mut child2 = Vec::with_capacity(len);

                for i in 0..len {
                    let p1 = parent1.genes()[i];
                    let p2 = parent2.genes()[i];

                    let u: f64 = rng.gen();
                    let beta = if u <= 0.5 {
                        (2.0 * u).powf(1.0 / (eta + 1.0))
                    } else {
                        (1.0 / (2.0 * (1.0 - u))).powf(1.0 / (eta + 1.0))
                    };

                    child1.push(0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2));
                    child2.push(0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2));
                }

                (RealGenome::new(child1), RealGenome::new(child2))
            }

            Self::Arithmetic => {
                let alpha: f64 = rng.gen();
                let mut child1 = Vec::with_capacity(len);
                let mut child2 = Vec::with_capacity(len);

                for i in 0..len {
                    let p1 = parent1.genes()[i];
                    let p2 = parent2.genes()[i];
                    child1.push(alpha * p1 + (1.0 - alpha) * p2);
                    child2.push((1.0 - alpha) * p1 + alpha * p2);
                }

                (RealGenome::new(child1), RealGenome::new(child2))
            }
        }
    }
}

impl Crossover<BinaryGenome> for BinaryCrossover {
    fn crossover<R: Rng>(
        &self,
        parent1: &BinaryGenome,
        parent2: &BinaryGenome,
        rng: &mut R,
    ) -> (BinaryGenome, BinaryGenome) {
        let len = parent1.len().min(parent2.len());
        if len == 0 {
            return (BinaryGenome::default(), BinaryGenome::default());
        }

        match self {
            Self::SinglePoint => {
                let point = rng.gen_range(0..=len);
                let mut child1: BitVec<u64, Lsb0> = BitVec::with_capacity(len);
                let mut child2: BitVec<u64, Lsb0> = BitVec::with_capacity(len);

                for i in 0..len {
                    if i < point {
                        child1.push(parent1.bits()[i]);
                        child2.push(parent2.bits()[i]);
                    } else {
                        child1.push(parent2.bits()[i]);
                        child2.push(parent1.bits()[i]);
                    }
                }

                (BinaryGenome::new(child1), BinaryGenome::new(child2))
            }

            Self::TwoPoint => {
                let mut p1 = rng.gen_range(0..=len);
                let mut p2 = rng.gen_range(0..=len);
                if p1 > p2 {
                    std::mem::swap(&mut p1, &mut p2);
                }

                let mut child1: BitVec<u64, Lsb0> = BitVec::with_capacity(len);
                let mut child2: BitVec<u64, Lsb0> = BitVec::with_capacity(len);

                for i in 0..len {
                    if i < p1 || i >= p2 {
                        child1.push(parent1.bits()[i]);
                        child2.push(parent2.bits()[i]);
                    } else {
                        child1.push(parent2.bits()[i]);
                        child2.push(parent1.bits()[i]);
                    }
                }

                (BinaryGenome::new(child1), BinaryGenome::new(child2))
            }

            Self::Uniform(prob) => {
                let mut child1: BitVec<u64, Lsb0> = BitVec::with_capacity(len);
                let mut child2: BitVec<u64, Lsb0> = BitVec::with_capacity(len);

                for i in 0..len {
                    if rng.gen::<f64>() < *prob {
                        child1.push(parent2.bits()[i]);
                        child2.push(parent1.bits()[i]);
                    } else {
                        child1.push(parent1.bits()[i]);
                        child2.push(parent2.bits()[i]);
                    }
                }

                (BinaryGenome::new(child1), BinaryGenome::new(child2))
            }
        }
    }
}

impl Crossover<PermutationGenome> for PermutationCrossover {
    fn crossover<R: Rng>(
        &self,
        parent1: &PermutationGenome,
        parent2: &PermutationGenome,
        rng: &mut R,
    ) -> (PermutationGenome, PermutationGenome) {
        let len = parent1.len().min(parent2.len());
        if len < 2 {
            return (parent1.clone(), parent2.clone());
        }

        match self {
            Self::Order => order_crossover(parent1, parent2, rng),
            Self::PartiallyMapped => pmx_crossover(parent1, parent2, rng),
            Self::Cycle => cycle_crossover(parent1, parent2),
            Self::EdgeRecombination => edge_recombination(parent1, parent2, rng),
        }
    }
}

/// Order Crossover (OX1) for permutations.
#[allow(clippy::items_after_statements)]
fn order_crossover<R: Rng>(
    parent1: &PermutationGenome,
    parent2: &PermutationGenome,
    rng: &mut R,
) -> (PermutationGenome, PermutationGenome) {
    let len = parent1.len();
    let mut p1 = rng.gen_range(0..len);
    let mut p2 = rng.gen_range(0..len);
    if p1 > p2 {
        std::mem::swap(&mut p1, &mut p2);
    }

    let mut child1 = vec![usize::MAX; len];
    let mut child2 = vec![usize::MAX; len];

    // Copy the segment
    for i in p1..=p2 {
        child1[i] = parent1[i];
        child2[i] = parent2[i];
    }

    // Fill remaining positions
    fn fill_remaining(child: &mut [usize], parent: &PermutationGenome, _p1: usize, p2: usize) {
        let len = child.len();
        let mut pos = (p2 + 1) % len;

        for i in 0..len {
            let idx = (p2 + 1 + i) % len;
            let gene = parent[idx];

            if !child.contains(&gene) {
                child[pos] = gene;
                pos = (pos + 1) % len;
            }
        }
    }

    fill_remaining(&mut child1, parent2, p1, p2);
    fill_remaining(&mut child2, parent1, p1, p2);

    let mut g1 = PermutationGenome::new(child1);
    let mut g2 = PermutationGenome::new(child2);
    g1.repair();
    g2.repair();

    (g1, g2)
}

/// Partially Mapped Crossover (PMX) for permutations.
#[allow(clippy::items_after_statements)]
fn pmx_crossover<R: Rng>(
    parent1: &PermutationGenome,
    parent2: &PermutationGenome,
    rng: &mut R,
) -> (PermutationGenome, PermutationGenome) {
    let len = parent1.len();
    let mut p1 = rng.gen_range(0..len);
    let mut p2 = rng.gen_range(0..len);
    if p1 > p2 {
        std::mem::swap(&mut p1, &mut p2);
    }

    let mut child1 = vec![usize::MAX; len];
    let mut child2 = vec![usize::MAX; len];

    // Copy segments
    for i in p1..=p2 {
        child1[i] = parent1[i];
        child2[i] = parent2[i];
    }

    // Build mapping
    let mut map1: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut map2: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();

    for i in p1..=p2 {
        map1.insert(parent2[i], parent1[i]);
        map2.insert(parent1[i], parent2[i]);
    }

    // Fill remaining positions
    fn fill_position(
        child: &mut [usize],
        parent: &PermutationGenome,
        map: &std::collections::HashMap<usize, usize>,
        p1: usize,
        p2: usize,
    ) {
        let len = child.len();
        for i in 0..len {
            if i >= p1 && i <= p2 {
                continue;
            }

            let mut gene = parent[i];
            while child[p1..=p2].contains(&gene) {
                gene = *map.get(&gene).unwrap_or(&gene);
            }
            child[i] = gene;
        }
    }

    fill_position(&mut child1, parent2, &map1, p1, p2);
    fill_position(&mut child2, parent1, &map2, p1, p2);

    let mut g1 = PermutationGenome::new(child1);
    let mut g2 = PermutationGenome::new(child2);
    g1.repair();
    g2.repair();

    (g1, g2)
}

/// Cycle Crossover (CX) for permutations.
fn cycle_crossover(
    parent1: &PermutationGenome,
    parent2: &PermutationGenome,
) -> (PermutationGenome, PermutationGenome) {
    let len = parent1.len();
    let mut child1 = vec![usize::MAX; len];
    let mut child2 = vec![usize::MAX; len];
    let mut cycle_num = 0;

    let pos1: std::collections::HashMap<usize, usize> = parent1
        .order()
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i))
        .collect();

    let mut used = vec![false; len];

    for start in 0..len {
        if used[start] {
            continue;
        }

        // Find cycle
        let mut cycle = Vec::new();
        let mut pos = start;

        loop {
            cycle.push(pos);
            used[pos] = true;
            let val = parent2[pos];
            pos = *pos1.get(&val).unwrap();

            if pos == start {
                break;
            }
        }

        // Alternate which parent each cycle comes from
        for &idx in &cycle {
            if cycle_num % 2 == 0 {
                child1[idx] = parent1[idx];
                child2[idx] = parent2[idx];
            } else {
                child1[idx] = parent2[idx];
                child2[idx] = parent1[idx];
            }
        }

        cycle_num += 1;
    }

    (
        PermutationGenome::new(child1),
        PermutationGenome::new(child2),
    )
}

/// Edge Recombination Crossover for permutations.
#[allow(clippy::items_after_statements)]
fn edge_recombination<R: Rng>(
    parent1: &PermutationGenome,
    parent2: &PermutationGenome,
    rng: &mut R,
) -> (PermutationGenome, PermutationGenome) {
    // Simplified version: just use order crossover for second child
    let len = parent1.len();

    // Build edge table
    let mut edges: std::collections::HashMap<usize, std::collections::HashSet<usize>> =
        std::collections::HashMap::new();

    for i in 0..len {
        edges.insert(i, std::collections::HashSet::new());
    }

    fn add_edges(
        edges: &mut std::collections::HashMap<usize, std::collections::HashSet<usize>>,
        parent: &PermutationGenome,
    ) {
        let len = parent.len();
        for i in 0..len {
            let curr = parent[i];
            let prev = parent[(i + len - 1) % len];
            let next = parent[(i + 1) % len];
            edges.get_mut(&curr).unwrap().insert(prev);
            edges.get_mut(&curr).unwrap().insert(next);
        }
    }

    add_edges(&mut edges, parent1);
    add_edges(&mut edges, parent2);

    // Build child
    let mut child = Vec::with_capacity(len);
    let mut current = parent1[0];
    let mut remaining: std::collections::HashSet<usize> = (0..len).collect();

    while child.len() < len {
        child.push(current);
        remaining.remove(&current);

        // Remove current from all edge lists
        for edge_set in edges.values_mut() {
            edge_set.remove(&current);
        }

        if remaining.is_empty() {
            break;
        }

        // Choose next: prefer neighbor with fewest edges
        let neighbors = edges.get(&current).cloned().unwrap_or_default();
        let valid_neighbors: Vec<_> = neighbors.iter().filter(|n| remaining.contains(n)).collect();

        current = if valid_neighbors.is_empty() {
            // Pick random remaining
            *remaining.iter().next().unwrap()
        } else {
            // Pick neighbor with fewest edges
            **valid_neighbors
                .iter()
                .min_by_key(|&&n| {
                    edges
                        .get(n)
                        .map_or(usize::MAX, std::collections::HashSet::len)
                })
                .unwrap()
        };
    }

    let child1 = PermutationGenome::new(child);
    let child2 = order_crossover(parent1, parent2, rng).1;

    (child1, child2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_real_single_point() {
        let p1 = RealGenome::new(vec![1.0, 2.0, 3.0, 4.0]);
        let p2 = RealGenome::new(vec![5.0, 6.0, 7.0, 8.0]);
        let mut rng = crate::rng::create_rng(Some(42));

        let (c1, c2) = RealCrossover::SinglePoint.crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 4);
        assert_eq!(c2.len(), 4);
    }

    #[test]
    fn test_blend_crossover() {
        let p1 = RealGenome::new(vec![0.0, 0.0]);
        let p2 = RealGenome::new(vec![10.0, 10.0]);
        let mut rng = crate::rng::create_rng(Some(42));

        let (c1, c2) = RealCrossover::Blend(0.5).crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 2);
        assert_eq!(c2.len(), 2);
    }

    #[test]
    fn test_binary_crossover() {
        let p1 = BinaryGenome::from_bools(&[true, true, true, true]);
        let p2 = BinaryGenome::from_bools(&[false, false, false, false]);
        let mut rng = crate::rng::create_rng(Some(42));

        let (c1, c2) = BinaryCrossover::TwoPoint.crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 4);
        assert_eq!(c2.len(), 4);
    }

    #[test]
    fn test_order_crossover() {
        let p1 = PermutationGenome::new(vec![0, 1, 2, 3, 4]);
        let p2 = PermutationGenome::new(vec![4, 3, 2, 1, 0]);
        let mut rng = crate::rng::create_rng(Some(42));

        let (c1, c2) = PermutationCrossover::Order.crossover(&p1, &p2, &mut rng);
        assert_eq!(c1.len(), 5);
        assert_eq!(c2.len(), 5);

        // Verify valid permutations
        let mut sorted1: Vec<_> = c1.order().to_vec();
        let mut sorted2: Vec<_> = c2.order().to_vec();
        sorted1.sort_unstable();
        sorted2.sort_unstable();
        assert_eq!(sorted1, vec![0, 1, 2, 3, 4]);
        assert_eq!(sorted2, vec![0, 1, 2, 3, 4]);
    }
}
