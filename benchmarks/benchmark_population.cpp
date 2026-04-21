#include <benchmark/benchmark.h>
#include "benchmark_utils.hpp"
#include "neat/config.hpp"
#include "neat/population.hpp"
#include "neat/network.hpp"

using namespace neat::benchmark_utils;

static void BM_Population2000Generations(benchmark::State& state) {
    neat::Config cfg = get_standard_config();
    
    for (auto _ : state) {
        state.PauseTiming();
        neat::Population pop(cfg);
        state.ResumeTiming();

        for (int i = 0; i < 2000; ++i) {
            pop.run_generation(dummy_eval);
        }
    }
}

// Benchmark 1: Wall time to reach a target fitness
static void BM_WallTimeToTargetFitness(benchmark::State& state) {
    neat::Config cfg = get_snake_config(); 
    const double target_score = 100000.0; 

    for (auto _ : state) {
        state.PauseTiming();
        neat::Population pop(cfg);
        state.ResumeTiming();

        pop.run_until([&pop](neat::Network& net) {
            return snake_eval(net, pop.generation());
        }, [target_score](const neat::GenerationResult& r) {
            return r.best_fitness >= target_score || r.generation >= 1000; 
        });
    }
}

// Benchmark 2: Number of generations to reach a target fitness
static void BM_GenerationsToTargetFitness(benchmark::State& state) {
    neat::Config cfg = get_snake_config();
    const double target_score = 100000.0;

    double total_generations = 0;

    for (auto _ : state) {
        state.PauseTiming();
        neat::Population pop(cfg);
        state.ResumeTiming();

        auto run = pop.run_until([&pop](neat::Network& net) {
            return snake_eval(net, pop.generation());
        }, [target_score](const neat::GenerationResult& r) {
            return r.best_fitness >= target_score || r.generation >= 1000;
        });

        total_generations += run.generations.back().generation;
    }
    
    state.counters["AvgGenerations"] = benchmark::Counter(total_generations, benchmark::Counter::kAvgIterations);
}

BENCHMARK(BM_Population2000Generations)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(5);

BENCHMARK(BM_WallTimeToTargetFitness)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(10);

BENCHMARK(BM_GenerationsToTargetFitness)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(10);
