use crate::{Error, Result};

pub(crate) fn validate_rate(name: &str, value: f64) -> Result<()> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(Error::Config(format!(
            "{name} must be finite and in [0, 1]"
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_common(
    population_size: usize,
    genome_length: usize,
    elitism: usize,
    tournament_size: usize,
    mutation_rate: f64,
    crossover_rate: f64,
    lower_bounds: Option<&[f64]>,
    upper_bounds: Option<&[f64]>,
) -> Result<()> {
    if population_size == 0 {
        return Err(Error::Config(
            "population size must be greater than zero".into(),
        ));
    }
    if genome_length == 0 {
        return Err(Error::Config(
            "genome_length must be greater than zero".into(),
        ));
    }
    if elitism > population_size {
        return Err(Error::Config(
            "elitism must not exceed population size".into(),
        ));
    }
    if tournament_size == 0 || tournament_size > population_size {
        return Err(Error::Config(
            "tournament_size must be between 1 and population size".into(),
        ));
    }
    validate_rate("mutation_rate", mutation_rate)?;
    validate_rate("crossover_rate", crossover_rate)?;

    if let Some(bounds) = lower_bounds {
        validate_bounds_length("lower_bounds", bounds, genome_length)?;
    }
    if let Some(bounds) = upper_bounds {
        validate_bounds_length("upper_bounds", bounds, genome_length)?;
    }
    let defaults_lower = vec![-10.0; genome_length];
    let defaults_upper = vec![10.0; genome_length];
    let lower = lower_bounds.unwrap_or(&defaults_lower);
    let upper = upper_bounds.unwrap_or(&defaults_upper);
    for (index, (&low, &high)) in lower.iter().zip(upper).enumerate() {
        if !low.is_finite() || !high.is_finite() {
            return Err(Error::Config(format!(
                "bounds at index {index} must be finite"
            )));
        }
        if low > high {
            return Err(Error::Config(format!(
                "lower bound at index {index} must not exceed upper bound"
            )));
        }
    }
    Ok(())
}

fn validate_bounds_length(name: &str, bounds: &[f64], genome_length: usize) -> Result<()> {
    if bounds.len() != genome_length {
        return Err(Error::Config(format!(
            "{name} length must match genome_length ({genome_length})"
        )));
    }
    Ok(())
}
