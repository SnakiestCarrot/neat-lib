#pragma once

#include "neat/config.hpp"
#include "neat/genome.hpp"
#include "neat/innovation.hpp"
#include "neat/network.hpp"
#include "neat/random.hpp"

#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

namespace neat {

// Summary of one completed generation, returned by run_generation().
struct GenerationResult {
    uint32_t generation;
    double   best_fitness;
    double   mean_fitness;
    double   worst_fitness;
    uint32_t num_species;
    // Wall-clock timings in milliseconds for each phase.
    double   time_eval_ms;
    double   time_speciate_ms;
    double   time_reproduce_ms;
    double   time_total_ms;
};

// Result of a full run_until() call.
struct RunResult {
    std::vector<GenerationResult> generations;
    bool converged = false; // true if stop_fn returned true before the budget ran out
};

// ---------------------------------------------------------------------------
// Population
//
// Manages a population of genomes across generations.
//
// Basic usage:
//
//   neat::Config cfg;
//   cfg.num_inputs  = 4;
//   cfg.num_outputs = 2;
//
//   neat::Population pop(cfg);
//
//   auto result = pop.run_generation([](neat::Network& net) {
//       return evaluate(net);   // call net.activate({...}) and return a score
//   });
//
// Or to run until a condition is met:
//
//   auto result = pop.run_until(
//       [](neat::Network& net) { return evaluate(net); },
//       [](const neat::GenerationResult& r) {
//           return r.best_fitness >= 1000.0 || r.generation >= 500;
//       }
//   );
// ---------------------------------------------------------------------------
class Population {
public:
    // Function the user provides to evaluate a single genome.
    // Receives a ready-to-use Network. Should return a non-negative fitness score.
    using EvalFn = std::function<double(Network&)>;

    // Predicate the user provides to stop run_until().
    // Receives the result of the most recently completed generation.
    using StopFn = std::function<bool(const GenerationResult&)>;

    explicit Population(const Config& cfg);

    // -----------------------------------------------------------------------
    // Core API
    // -----------------------------------------------------------------------

    // Evaluate every genome in the current generation, then advance to the
    // next (speciate, compute adjusted fitness, reproduce).
    // Returns a summary of the generation that was just evaluated.
    GenerationResult run_generation(const EvalFn& eval_fn);

    // Keep calling run_generation() until stop_fn returns true.
    // Returns all per-generation results and a convergence flag.
    RunResult run_until(const EvalFn& eval_fn, const StopFn& stop_fn);

    // -----------------------------------------------------------------------
    // Visualization helpers
    //
    // These return Networks built from snapshots taken during the most recent
    // run_generation() call. Throws std::logic_error if no generation has run.
    // -----------------------------------------------------------------------

    Network best_network()   const; // Genome with the highest fitness
    Network worst_network()  const; // Genome with the lowest fitness
    Network random_network();       // A randomly chosen genome

    // -----------------------------------------------------------------------
    // Inspection
    // -----------------------------------------------------------------------

    uint32_t generation()   const { return generation_; }
    uint32_t num_species()  const { return static_cast<uint32_t>(species_.size()); }

    // Direct access to the current generation's genomes.
    // Intended for advanced use cases — prefer run_generation() for the
    // standard evolutionary loop.
    std::vector<Genome>& genomes();

private:
    struct Species {
        Genome              representative;
        std::vector<Genome> members;
        double              best_fitness = 0.0;
        uint32_t            stagnation   = 0;
    };

    void   speciate();
    void   adjust_fitness();
    void   reproduce();
    size_t best_species_index() const;

    Config            cfg_;
    Random            rng_;
    InnovationTracker innovations_;
    std::vector<Genome>  genomes_;
    std::vector<Species> species_;
    uint32_t             generation_ = 0;

    // Snapshots from the most recently evaluated generation.
    // Set by run_generation(), used by the visualization helpers.
    std::optional<Genome> snapshot_best_;
    std::optional<Genome> snapshot_worst_;
    std::optional<Genome> snapshot_random_;
};

} // namespace neat
