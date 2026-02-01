//! Performance benchmarks for the parga library.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use parga::prelude::*;

fn benchmark_ga_sphere(c: &mut Criterion) {
    let mut group = c.benchmark_group("ga_sphere");

    for pop_size in [50, 100, 200, 500].iter() {
        group.throughput(Throughput::Elements(*pop_size as u64));
        group.bench_with_input(
            BenchmarkId::new("population", pop_size),
            pop_size,
            |b, &size| {
                b.iter(|| {
                    let config = GaConfig::builder()
                        .population_size(size)
                        .genome_length(10)
                        .generations(50)
                        .seed(42)
                        .build()
                        .unwrap();

                    let fitness = Sphere;
                    let mut ga: GeneticAlgorithm<RealGenome, _> =
                        GeneticAlgorithm::new(config, fitness);
                    black_box(ga.run())
                });
            },
        );
    }

    group.finish();
}

fn benchmark_ga_dimensions(c: &mut Criterion) {
    let mut group = c.benchmark_group("ga_dimensions");

    for dims in [5, 10, 20, 50].iter() {
        group.bench_with_input(BenchmarkId::new("dimensions", dims), dims, |b, &d| {
            b.iter(|| {
                let config = GaConfig::builder()
                    .population_size(100)
                    .genome_length(d)
                    .generations(50)
                    .seed(42)
                    .build()
                    .unwrap();

                let fitness = Sphere;
                let mut ga: GeneticAlgorithm<RealGenome, _> =
                    GeneticAlgorithm::new(config, fitness);
                black_box(ga.run())
            });
        });
    }

    group.finish();
}

fn benchmark_island_model(c: &mut Criterion) {
    let mut group = c.benchmark_group("island_model");

    for num_islands in [2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("islands", num_islands),
            num_islands,
            |b, &n| {
                b.iter(|| {
                    let config = IslandConfig::builder()
                        .num_islands(n)
                        .island_population(50)
                        .genome_length(10)
                        .generations(30)
                        .migration_interval(10)
                        .seed(42)
                        .build()
                        .unwrap();

                    let fitness = Sphere;
                    let mut model: IslandModel<RealGenome, _> = IslandModel::new(config, fitness);
                    black_box(model.run())
                });
            },
        );
    }

    group.finish();
}

fn benchmark_selection_methods(c: &mut Criterion) {
    let mut group = c.benchmark_group("selection");

    let methods = [
        ("tournament", SelectionOperator::Tournament(3)),
        ("roulette", SelectionOperator::RouletteWheel),
        ("rank", SelectionOperator::Rank),
        ("truncation", SelectionOperator::Truncation(0.5)),
    ];

    for (name, method) in methods.iter() {
        group.bench_function(*name, |b| {
            b.iter(|| {
                let config = GaConfig::builder()
                    .population_size(100)
                    .genome_length(10)
                    .generations(30)
                    .seed(42)
                    .build()
                    .unwrap();

                let fitness = Sphere;
                let mut ga: GeneticAlgorithm<RealGenome, _> =
                    GeneticAlgorithm::new(config, fitness).with_selection(*method);
                black_box(ga.run())
            });
        });
    }

    group.finish();
}

fn benchmark_fitness_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("fitness_functions");

    // Create a test genome
    let genome = RealGenome::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

    group.bench_function("sphere", |b| b.iter(|| black_box(Sphere.evaluate(&genome))));

    group.bench_function("rastrigin", |b| {
        b.iter(|| black_box(Rastrigin.evaluate(&genome)))
    });

    group.bench_function("rosenbrock", |b| {
        b.iter(|| black_box(Rosenbrock.evaluate(&genome)))
    });

    group.bench_function("ackley", |b| b.iter(|| black_box(Ackley.evaluate(&genome))));

    group.bench_function("griewank", |b| {
        b.iter(|| black_box(Griewank.evaluate(&genome)))
    });

    group.finish();
}

fn benchmark_migration_topologies(c: &mut Criterion) {
    let mut group = c.benchmark_group("migration_topology");

    let topologies = [
        ("ring", MigrationTopology::Ring),
        ("star", MigrationTopology::Star),
        ("ladder", MigrationTopology::Ladder),
        ("fully_connected", MigrationTopology::FullyConnected),
    ];

    for (name, topology) in topologies.iter() {
        group.bench_function(*name, |b| {
            b.iter(|| {
                let config = IslandConfig::builder()
                    .num_islands(4)
                    .island_population(30)
                    .genome_length(10)
                    .generations(20)
                    .migration_interval(5)
                    .topology(*topology)
                    .seed(42)
                    .build()
                    .unwrap();

                let fitness = Sphere;
                let mut model: IslandModel<RealGenome, _> = IslandModel::new(config, fitness);
                black_box(model.run())
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_ga_sphere,
    benchmark_ga_dimensions,
    benchmark_island_model,
    benchmark_selection_methods,
    benchmark_fitness_functions,
    benchmark_migration_topologies,
);

criterion_main!(benches);
