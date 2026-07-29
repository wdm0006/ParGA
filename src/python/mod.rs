//! Python bindings for the parga library.
//!
//! This module provides PyO3 bindings to expose the genetic algorithm
//! functionality to Python.

use std::sync::{Arc, Mutex};

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

#[cfg(Py_GIL_DISABLED)]
use rayon::prelude::*;

use crate::fitness::FitnessFunction;
use crate::genome::RealGenome;
use crate::island::{IslandConfig, IslandModel, MigrationTopology};
use crate::operators::{
    crossover::{CrossoverOperator, RealCrossover},
    mutation::{MutationOperator, RealMutation},
    selection::SelectionOperator,
};
use crate::{GaConfig, GaResult, GeneticAlgorithm};

/// Configure the rayon global thread pool size.
///
/// Must be called before any parallel work is submitted. Calling after
/// the pool has been initialized will return an error.
#[pyfunction]
#[pyo3(signature = (num_threads))]
fn set_num_threads(num_threads: usize) -> PyResult<()> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Return whether the extension was compiled for free-threaded Python.
///
/// When `True`, Rust-side fitness evaluation uses rayon parallelism
/// instead of the Python `ProcessPoolExecutor` path.
#[pyfunction]
fn is_free_threaded() -> bool {
    cfg!(Py_GIL_DISABLED)
}

/// Register the Python module.
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGeneticAlgorithm>()?;
    m.add_class::<PyIslandModel>()?;
    m.add_class::<PyGaResult>()?;
    m.add_class::<PyIslandResult>()?;
    m.add_class::<PySelectionMethod>()?;
    m.add_class::<PyCrossoverMethod>()?;
    m.add_class::<PyMutationMethod>()?;
    m.add_class::<PyMigrationTopology>()?;

    // Add utility functions
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(is_free_threaded, m)?)?;

    // Add benchmark functions
    m.add_function(wrap_pyfunction!(sphere, m)?)?;
    m.add_function(wrap_pyfunction!(rastrigin, m)?)?;
    m.add_function(wrap_pyfunction!(rosenbrock, m)?)?;
    m.add_function(wrap_pyfunction!(ackley, m)?)?;
    m.add_function(wrap_pyfunction!(griewank, m)?)?;
    m.add_function(wrap_pyfunction!(schwefel, m)?)?;

    Ok(())
}

/// Python wrapper for fitness function that calls back to Python.
struct PyFitness {
    callback: PyObject,
    /// First error raised by the callback, if any. Recording a `String`
    /// (rather than a `PyErr`) keeps the slot `Send` for use under
    /// `py.allow_threads`. Surfaced to the caller after the run completes.
    error: Arc<Mutex<Option<String>>>,
}

impl PyFitness {
    fn new(callback: PyObject) -> Self {
        Self {
            callback,
            error: Arc::new(Mutex::new(None)),
        }
    }

    /// Record the first callback failure; later failures are ignored.
    fn record_error(&self, message: String) {
        let mut slot = self.error.lock().unwrap();
        if slot.is_none() {
            *slot = Some(message);
        }
    }

    /// Evaluate the callback for a single genome, recording any failure.
    fn call(&self, py: Python<'_>, array: Bound<'_, PyArray1<f64>>) -> f64 {
        let result = match self.callback.call1(py, (array,)) {
            Ok(result) => result,
            Err(err) => {
                self.record_error(err.to_string());
                return f64::NEG_INFINITY;
            }
        };

        let Ok(value) = result.extract::<f64>(py) else {
            self.record_error("fitness function did not return a float".to_string());
            return f64::NEG_INFINITY;
        };

        if !value.is_finite() {
            self.record_error(format!(
                "fitness function returned a non-finite value: {value}"
            ));
            return f64::NEG_INFINITY;
        }

        value
    }
}

impl Clone for PyFitness {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            callback: self.callback.clone_ref(py),
            error: Arc::clone(&self.error),
        })
    }
}

impl FitnessFunction<RealGenome> for PyFitness {
    fn evaluate(&self, genome: &RealGenome) -> f64 {
        Python::with_gil(|py| {
            let genes = genome.genes();
            let array = genes.to_vec().into_pyarray(py);
            self.call(py, array)
        })
    }

    fn evaluate_batch(&self, genomes: &[&RealGenome]) -> Vec<f64> {
        // Free-threaded Python (3.13t+): no GIL, so rayon par_iter can call
        // Python callbacks from multiple threads simultaneously.
        #[cfg(Py_GIL_DISABLED)]
        {
            genomes
                .par_iter()
                .map(|genome| self.evaluate(genome))
                .collect()
        }

        // GIL Python: acquire the GIL once for the entire batch.
        // Uses from_slice to avoid Vec allocation per genome.
        #[cfg(not(Py_GIL_DISABLED))]
        {
            Python::with_gil(|py| {
                genomes
                    .iter()
                    .map(|genome| {
                        let genes = genome.genes();
                        let array = PyArray1::from_slice(py, genes);
                        self.call(py, array)
                    })
                    .collect()
            })
        }
    }
}

/// Selection method enumeration for Python.
#[pyclass(name = "SelectionMethod")]
#[derive(Clone)]
pub struct PySelectionMethod {
    inner: SelectionOperator,
}

#[pymethods]
impl PySelectionMethod {
    /// Tournament selection.
    #[staticmethod]
    #[pyo3(signature = (size=3))]
    fn tournament(size: usize) -> Self {
        Self {
            inner: SelectionOperator::Tournament(size),
        }
    }

    /// Roulette wheel selection.
    #[staticmethod]
    fn roulette() -> Self {
        Self {
            inner: SelectionOperator::RouletteWheel,
        }
    }

    /// Rank-based selection.
    #[staticmethod]
    fn rank() -> Self {
        Self {
            inner: SelectionOperator::Rank,
        }
    }

    /// Random selection.
    #[staticmethod]
    fn random() -> Self {
        Self {
            inner: SelectionOperator::Random,
        }
    }

    /// Truncation selection.
    #[staticmethod]
    #[pyo3(signature = (ratio=0.5))]
    fn truncation(ratio: f64) -> Self {
        Self {
            inner: SelectionOperator::Truncation(ratio),
        }
    }

    /// Stochastic universal sampling.
    #[staticmethod]
    fn stochastic_universal() -> Self {
        Self {
            inner: SelectionOperator::StochasticUniversal,
        }
    }
}

/// Crossover method enumeration for Python.
#[pyclass(name = "CrossoverMethod")]
#[derive(Clone)]
pub struct PyCrossoverMethod {
    inner: RealCrossover,
}

#[pymethods]
impl PyCrossoverMethod {
    /// Single-point crossover.
    #[staticmethod]
    fn single_point() -> Self {
        Self {
            inner: RealCrossover::SinglePoint,
        }
    }

    /// Two-point crossover.
    #[staticmethod]
    fn two_point() -> Self {
        Self {
            inner: RealCrossover::TwoPoint,
        }
    }

    /// Uniform crossover.
    #[staticmethod]
    #[pyo3(signature = (probability=0.5))]
    fn uniform(probability: f64) -> Self {
        Self {
            inner: RealCrossover::Uniform(probability),
        }
    }

    /// Blend crossover (BLX-alpha).
    #[staticmethod]
    #[pyo3(signature = (alpha=0.5))]
    fn blend(alpha: f64) -> Self {
        Self {
            inner: RealCrossover::Blend(alpha),
        }
    }

    /// Simulated binary crossover (SBX).
    #[staticmethod]
    #[pyo3(signature = (eta=20.0))]
    fn simulated_binary(eta: f64) -> Self {
        Self {
            inner: RealCrossover::SimulatedBinary(eta),
        }
    }

    /// Arithmetic crossover.
    #[staticmethod]
    fn arithmetic() -> Self {
        Self {
            inner: RealCrossover::Arithmetic,
        }
    }
}

/// Mutation method enumeration for Python.
#[pyclass(name = "MutationMethod")]
#[derive(Clone)]
pub struct PyMutationMethod {
    inner: RealMutation,
}

#[pymethods]
impl PyMutationMethod {
    /// Gaussian mutation.
    #[staticmethod]
    #[pyo3(signature = (sigma=0.1))]
    fn gaussian(sigma: f64) -> Self {
        Self {
            inner: RealMutation::Gaussian(sigma),
        }
    }

    /// Uniform mutation.
    #[staticmethod]
    fn uniform() -> Self {
        Self {
            inner: RealMutation::Uniform,
        }
    }

    /// Polynomial mutation.
    #[staticmethod]
    #[pyo3(signature = (eta=20.0))]
    fn polynomial(eta: f64) -> Self {
        Self {
            inner: RealMutation::Polynomial(eta),
        }
    }

    /// Boundary mutation.
    #[staticmethod]
    fn boundary() -> Self {
        Self {
            inner: RealMutation::Boundary,
        }
    }
}

/// Migration topology for island model.
#[pyclass(name = "MigrationTopology")]
#[derive(Clone)]
pub struct PyMigrationTopology {
    inner: MigrationTopology,
}

#[pymethods]
impl PyMigrationTopology {
    /// Ring topology.
    #[staticmethod]
    fn ring() -> Self {
        Self {
            inner: MigrationTopology::Ring,
        }
    }

    /// Fully connected topology.
    #[staticmethod]
    fn fully_connected() -> Self {
        Self {
            inner: MigrationTopology::FullyConnected,
        }
    }

    /// Random topology.
    #[staticmethod]
    fn random() -> Self {
        Self {
            inner: MigrationTopology::Random,
        }
    }

    /// Star topology.
    #[staticmethod]
    fn star() -> Self {
        Self {
            inner: MigrationTopology::Star,
        }
    }

    /// Ladder topology.
    #[staticmethod]
    fn ladder() -> Self {
        Self {
            inner: MigrationTopology::Ladder,
        }
    }
}

/// Result of a genetic algorithm run.
#[pyclass(name = "GaResult")]
pub struct PyGaResult {
    #[pyo3(get)]
    best_fitness: f64,
    #[pyo3(get)]
    generations: usize,
    #[pyo3(get)]
    converged: bool,
    best_genes: Vec<f64>,
    fitness_history: Vec<f64>,
}

#[pymethods]
impl PyGaResult {
    /// Returns the best genes as a numpy array.
    fn best_genes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.best_genes.clone().into_pyarray(py)
    }

    /// Returns the fitness history as a numpy array.
    fn fitness_history<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.fitness_history.clone().into_pyarray(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "GaResult(best_fitness={:.6}, generations={}, converged={})",
            self.best_fitness, self.generations, self.converged
        )
    }
}

impl From<GaResult<RealGenome>> for PyGaResult {
    fn from(result: GaResult<RealGenome>) -> Self {
        Self {
            best_fitness: result.best_fitness,
            generations: result.generations,
            converged: result.converged,
            best_genes: result.best_individual.genome.genes().to_vec(),
            fitness_history: result.fitness_history,
        }
    }
}

/// Result of an island model run.
#[pyclass(name = "IslandResult")]
pub struct PyIslandResult {
    #[pyo3(get)]
    best_fitness: f64,
    #[pyo3(get)]
    generations: usize,
    #[pyo3(get)]
    converged: bool,
    best_genes: Vec<f64>,
    island_best_fitness: Vec<f64>,
    fitness_history: Vec<f64>,
}

#[pymethods]
impl PyIslandResult {
    /// Returns the best genes as a numpy array.
    fn best_genes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.best_genes.clone().into_pyarray(py)
    }

    /// Returns the best fitness per island as a numpy array.
    fn island_best_fitness<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.island_best_fitness.clone().into_pyarray(py)
    }

    /// Returns the fitness history as a numpy array.
    fn fitness_history<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.fitness_history.clone().into_pyarray(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "IslandResult(best_fitness={:.6}, generations={}, converged={})",
            self.best_fitness, self.generations, self.converged
        )
    }
}

/// Standard genetic algorithm.
#[pyclass(name = "GeneticAlgorithm")]
pub struct PyGeneticAlgorithm {
    config: GaConfig,
    fitness_fn: PyObject,
    selection: Option<SelectionOperator>,
    crossover: Option<RealCrossover>,
    mutation: Option<RealMutation>,
}

#[pymethods]
impl PyGeneticAlgorithm {
    /// Creates a new genetic algorithm.
    ///
    /// Args:
    ///     fitness_fn: A callable that takes a numpy array and returns a finite
    ///         float. NaN and +/-inf results are rejected with a RuntimeError.
    ///     genome_length: Length of each genome.
    ///     population_size: Number of individuals in the population.
    ///     generations: Number of generations to evolve.
    ///     mutation_rate: Probability of mutation per gene.
    ///     crossover_rate: Probability of crossover.
    ///     elitism: Number of elite individuals to preserve.
    ///     lower_bounds: Lower bounds for each gene (optional).
    ///     upper_bounds: Upper bounds for each gene (optional).
    ///     seed: Random seed for reproducibility (optional).
    #[new]
    #[pyo3(signature = (
        fitness_fn,
        genome_length,
        population_size = 100,
        generations = 100,
        mutation_rate = 0.01,
        crossover_rate = 0.8,
        elitism = 2,
        tournament_size = 3,
        lower_bounds = None,
        upper_bounds = None,
        seed = None,
        early_stopping = None,
        restart_on_stagnation = None,
        local_search_iters = None,
        mutation_rate_end = None,
        random_immigrants = None
    ))]
    fn new(
        fitness_fn: PyObject,
        genome_length: usize,
        population_size: usize,
        generations: usize,
        mutation_rate: f64,
        crossover_rate: f64,
        elitism: usize,
        tournament_size: usize,
        lower_bounds: Option<Vec<f64>>,
        upper_bounds: Option<Vec<f64>>,
        seed: Option<u64>,
        early_stopping: Option<usize>,
        restart_on_stagnation: Option<usize>,
        local_search_iters: Option<usize>,
        mutation_rate_end: Option<f64>,
        random_immigrants: Option<usize>,
    ) -> PyResult<Self> {
        let mut builder = GaConfig::builder();
        builder
            .population_size(population_size)
            .genome_length(genome_length)
            .generations(generations)
            .mutation_rate(mutation_rate)
            .crossover_rate(crossover_rate)
            .elitism(elitism)
            .tournament_size(tournament_size);

        if let Some(lb) = lower_bounds {
            builder.lower_bounds(lb);
        }
        if let Some(ub) = upper_bounds {
            builder.upper_bounds(ub);
        }
        if let Some(s) = seed {
            builder.seed(s);
        }
        if let Some(es) = early_stopping {
            builder.early_stopping(es);
        }
        if let Some(rs) = restart_on_stagnation {
            builder.restart_on_stagnation(rs);
        }
        if let Some(ls) = local_search_iters {
            builder.local_search_iters(ls);
        }
        if let Some(mre) = mutation_rate_end {
            builder.mutation_rate_end(mre);
        }
        if let Some(ri) = random_immigrants {
            builder.random_immigrants(ri);
        }

        let config = builder
            .build()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        config
            .validate()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(Self {
            config,
            fitness_fn,
            selection: None,
            crossover: None,
            mutation: None,
        })
    }

    /// Sets the selection method.
    fn set_selection(&mut self, method: &PySelectionMethod) {
        self.selection = Some(method.inner);
    }

    /// Sets the crossover method.
    fn set_crossover(&mut self, method: &PyCrossoverMethod) {
        self.crossover = Some(method.inner);
    }

    /// Sets the mutation method.
    fn set_mutation(&mut self, method: &PyMutationMethod) {
        self.mutation = Some(method.inner);
    }

    /// Runs the genetic algorithm.
    fn run(&self, py: Python<'_>) -> PyResult<PyGaResult> {
        let fitness = PyFitness::new(self.fitness_fn.clone_ref(py));
        let error_slot = Arc::clone(&fitness.error);

        // Allow other Python threads to run during evolution
        let result = py.allow_threads(|| {
            let mut ga: GeneticAlgorithm<RealGenome, PyFitness> =
                GeneticAlgorithm::new(self.config.clone(), fitness).unwrap();

            if let Some(selection) = self.selection {
                ga = ga.with_selection(selection);
            }

            if let Some(crossover) = self.crossover {
                ga = ga.with_crossover(CrossoverOperator::Real(crossover));
            }

            if let Some(mutation) = self.mutation {
                ga = ga.with_mutation(MutationOperator::Real(mutation));
            }

            ga.run()
        });

        if let Some(message) = error_slot.lock().unwrap().take() {
            return Err(PyRuntimeError::new_err(format!(
                "fitness function failed: {message}"
            )));
        }

        Ok(PyGaResult::from(result))
    }
}

/// Island model genetic algorithm.
#[pyclass(name = "IslandModel")]
pub struct PyIslandModel {
    config: IslandConfig,
    fitness_fn: PyObject,
    selection: Option<SelectionOperator>,
    crossover: Option<RealCrossover>,
    mutation: Option<RealMutation>,
}

#[pymethods]
impl PyIslandModel {
    /// Creates a new island model.
    ///
    /// Args:
    ///     fitness_fn: A callable that takes a numpy array and returns a finite
    ///         float. NaN and +/-inf results are rejected with a RuntimeError.
    ///     genome_length: Length of each genome.
    ///     num_islands: Number of islands.
    ///     island_population: Population size per island.
    ///     generations: Total generations to evolve.
    ///     migration_interval: Generations between migrations.
    ///     migration_count: Number of individuals to migrate.
    ///     topology: Migration topology (optional).
    ///     mutation_rate: Probability of mutation per gene.
    ///     crossover_rate: Probability of crossover.
    ///     elitism: Number of elite individuals per island.
    ///     lower_bounds: Lower bounds for each gene (optional).
    ///     upper_bounds: Upper bounds for each gene (optional).
    ///     seed: Random seed for reproducibility (optional).
    #[new]
    #[pyo3(signature = (
        fitness_fn,
        genome_length,
        num_islands = 4,
        island_population = 100,
        generations = 100,
        migration_interval = 10,
        migration_count = 5,
        topology = None,
        mutation_rate = 0.01,
        crossover_rate = 0.8,
        elitism = 2,
        tournament_size = 3,
        lower_bounds = None,
        upper_bounds = None,
        seed = None
    ))]
    fn new(
        fitness_fn: PyObject,
        genome_length: usize,
        num_islands: usize,
        island_population: usize,
        generations: usize,
        migration_interval: usize,
        migration_count: usize,
        topology: Option<&PyMigrationTopology>,
        mutation_rate: f64,
        crossover_rate: f64,
        elitism: usize,
        tournament_size: usize,
        lower_bounds: Option<Vec<f64>>,
        upper_bounds: Option<Vec<f64>>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let topo = topology.map(|t| t.inner).unwrap_or(MigrationTopology::Ring);

        let mut builder = IslandConfig::builder();
        builder
            .num_islands(num_islands)
            .island_population(island_population)
            .genome_length(genome_length)
            .generations(generations)
            .migration_interval(migration_interval)
            .migration_count(migration_count)
            .topology(topo)
            .mutation_rate(mutation_rate)
            .crossover_rate(crossover_rate)
            .elitism(elitism)
            .tournament_size(tournament_size);

        if let Some(lb) = lower_bounds {
            builder.lower_bounds(lb);
        }
        if let Some(ub) = upper_bounds {
            builder.upper_bounds(ub);
        }
        if let Some(s) = seed {
            builder.seed(s);
        }

        let config = builder
            .build()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        config
            .validate()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(Self {
            config,
            fitness_fn,
            selection: None,
            crossover: None,
            mutation: None,
        })
    }

    /// Sets the selection method.
    fn set_selection(&mut self, method: &PySelectionMethod) {
        self.selection = Some(method.inner);
    }

    /// Sets the crossover method.
    fn set_crossover(&mut self, method: &PyCrossoverMethod) {
        self.crossover = Some(method.inner);
    }

    /// Sets the mutation method.
    fn set_mutation(&mut self, method: &PyMutationMethod) {
        self.mutation = Some(method.inner);
    }

    /// Runs the island model.
    fn run(&self, py: Python<'_>) -> PyResult<PyIslandResult> {
        let fitness = PyFitness::new(self.fitness_fn.clone_ref(py));
        let error_slot = Arc::clone(&fitness.error);

        let result = py.allow_threads(|| {
            let mut model: IslandModel<RealGenome, PyFitness> =
                IslandModel::new(self.config.clone(), fitness).unwrap();

            if let Some(selection) = self.selection {
                model = model.with_selection(selection);
            }

            if let Some(crossover) = self.crossover {
                model = model.with_crossover(CrossoverOperator::Real(crossover));
            }

            if let Some(mutation) = self.mutation {
                model = model.with_mutation(MutationOperator::Real(mutation));
            }

            model.run()
        });

        if let Some(message) = error_slot.lock().unwrap().take() {
            return Err(PyRuntimeError::new_err(format!(
                "fitness function failed: {message}"
            )));
        }

        Ok(PyIslandResult {
            best_fitness: result.best_fitness,
            generations: result.generations,
            converged: result.converged,
            best_genes: result.best_individual.genome.genes().to_vec(),
            island_best_fitness: result.island_best_fitness,
            fitness_history: result.fitness_history,
        })
    }
}

// Benchmark functions exposed to Python

/// Sphere function (minimize sum of squares).
#[pyfunction]
fn sphere(x: PyReadonlyArray1<'_, f64>) -> f64 {
    let genes = x.as_array();
    -genes.iter().map(|v| v * v).sum::<f64>()
}

/// Rastrigin function.
#[pyfunction]
fn rastrigin(x: PyReadonlyArray1<'_, f64>) -> f64 {
    use std::f64::consts::PI;
    let genes = x.as_array();
    let n = genes.len() as f64;
    let sum: f64 = genes
        .iter()
        .map(|&v| v * v - 10.0 * (2.0 * PI * v).cos())
        .sum();
    -(10.0 * n + sum)
}

/// Rosenbrock function.
#[pyfunction]
fn rosenbrock(x: PyReadonlyArray1<'_, f64>) -> f64 {
    let genes = x.as_array();
    if genes.len() < 2 {
        return 0.0;
    }
    let sum: f64 = genes
        .iter()
        .zip(genes.iter().skip(1))
        .map(|(&a, &b)| 100.0 * (b - a * a).powi(2) + (1.0 - a).powi(2))
        .sum();
    -sum
}

/// Ackley function.
#[pyfunction]
fn ackley(x: PyReadonlyArray1<'_, f64>) -> f64 {
    use std::f64::consts::{E, PI};
    let genes = x.as_array();
    let n = genes.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let sum_sq: f64 = genes.iter().map(|v| v * v).sum();
    let sum_cos: f64 = genes.iter().map(|v| (2.0 * PI * v).cos()).sum();
    let term1 = -20.0 * (-0.2 * (sum_sq / n).sqrt()).exp();
    let term2 = -(sum_cos / n).exp();
    -(term1 + term2 + 20.0 + E)
}

/// Griewank function.
#[pyfunction]
fn griewank(x: PyReadonlyArray1<'_, f64>) -> f64 {
    let genes = x.as_array();
    let sum: f64 = genes.iter().map(|v| v * v / 4000.0).sum();
    let prod: f64 = genes
        .iter()
        .enumerate()
        .map(|(i, &v)| (v / ((i + 1) as f64).sqrt()).cos())
        .product();
    -(sum - prod + 1.0)
}

/// Schwefel function.
#[pyfunction]
fn schwefel(x: PyReadonlyArray1<'_, f64>) -> f64 {
    let genes = x.as_array();
    let n = genes.len() as f64;
    let sum: f64 = genes.iter().map(|&v| v * v.abs().sqrt().sin()).sum();
    -(418.9829 * n - sum)
}
