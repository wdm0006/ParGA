//! Error types for the parga library.

use thiserror::Error;

/// Result type alias for parga operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Error types that can occur in the parga library.
#[derive(Error, Debug, Clone)]
pub enum Error {
    /// Configuration error.
    #[error("Configuration error: {0}")]
    Config(String),

    /// Invalid genome error.
    #[error("Invalid genome: {0}")]
    InvalidGenome(String),

    /// Invalid bounds error.
    #[error("Invalid bounds: {0}")]
    InvalidBounds(String),

    /// Population error.
    #[error("Population error: {0}")]
    Population(String),

    /// Island model error.
    #[error("Island model error: {0}")]
    Island(String),

    /// Operator error.
    #[error("Operator error: {0}")]
    Operator(String),

    /// Fitness function error.
    #[error("Fitness function error: {0}")]
    Fitness(String),

    /// Empty population error.
    #[error("Population is empty")]
    EmptyPopulation,

    /// Dimension mismatch error.
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
}
