//! Random number generator utilities.

use rand::{rngs::StdRng, SeedableRng};

/// Creates a random number generator, optionally seeded.
pub fn create_rng(seed: Option<u64>) -> StdRng {
    match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    }
}

/// Creates a thread-local random number generator.
pub fn thread_rng() -> impl rand::Rng {
    rand::thread_rng()
}
