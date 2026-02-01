//! Genetic operators: selection, crossover, and mutation.
//!
//! This module provides the core operators for genetic algorithms:
//! - [`selection`]: Methods for selecting parents (tournament, roulette, elitism)
//! - [`crossover`]: Methods for combining parents (single-point, two-point, uniform)
//! - [`mutation`]: Methods for introducing variation (gaussian, swap, flip)

pub mod crossover;
pub mod mutation;
pub mod selection;

pub use crossover::{Crossover, CrossoverOperator};
pub use mutation::{Mutation, MutationOperator};
pub use selection::{Selection, SelectionOperator};
