#include "neat/population.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <stdexcept>

#if defined(NEAT_PARALLEL_EVAL)
    #include <execution>
#endif

namespace neat {

// ===========================================================================
// Construction
// ===========================================================================

Population::Population(const Config& cfg)
    : cfg_(cfg)
    , rng_(cfg.seed)
    , innovations_()
{
    cfg_.validate();

    genomes_.reserve(cfg_.population_size);
    for (uint32_t i = 0; i < cfg_.population_size; ++i) {
        genomes_.push_back(Genome::create_minimal(
            cfg_.num_inputs, cfg_.num_outputs, rng_, innovations_
        ));
    }
}

std::vector<Genome>& Population::genomes() {
    return genomes_;
}

// ===========================================================================
// Public API — run_generation / run_until / visualization helpers
// ===========================================================================

GenerationResult Population::run_generation(const EvalFn& eval_fn) {
    using Clock = std::chrono::steady_clock;
    using Ms    = std::chrono::duration<double, std::milli>;

    auto t_total_start = Clock::now();

    // --- Evaluation phase ---------------------------------------------------
    // Each genome is independent: Network construction and eval_fn have no
    // shared mutable state, so this loop is safe to parallelise.
    // The speciate/reproduce phases always run single-threaded, so the
    // evolutionary outcome is identical regardless of parallel_eval.
    auto eval_one = [&](Genome& genome) {
        Network net(genome, cfg_);
        genome.fitness = eval_fn(net);
    };

    auto t_eval_start = Clock::now();
#if defined(NEAT_PARALLEL_EVAL)
    if (cfg_.parallel_eval) {
        std::for_each(std::execution::par_unseq,
                      genomes_.begin(), genomes_.end(), eval_one);
    } else {
        std::for_each(genomes_.begin(), genomes_.end(), eval_one);
    }
#else
    std::for_each(genomes_.begin(), genomes_.end(), eval_one);
#endif
    double time_eval_ms = Ms(Clock::now() - t_eval_start).count();

    // Take snapshots before speciate/reproduce replace genomes_.
    auto best_it  = std::max_element(genomes_.begin(), genomes_.end(),
        [](const Genome& a, const Genome& b) { return a.fitness < b.fitness; });
    auto worst_it = std::min_element(genomes_.begin(), genomes_.end(),
        [](const Genome& a, const Genome& b) { return a.fitness < b.fitness; });
    int random_idx = rng_.random_int(0, static_cast<int>(genomes_.size()) - 1);

    snapshot_best_   = *best_it;
    snapshot_worst_  = *worst_it;
    snapshot_random_ = genomes_[random_idx];

    double total = 0.0;
    for (const auto& g : genomes_) total += g.fitness;
    uint32_t evaluated_generation = generation_;
    double   best_fitness         = best_it->fitness;
    double   mean_fitness         = total / static_cast<double>(genomes_.size());
    double   worst_fitness        = worst_it->fitness;

    // --- Speciation / selection phase ---------------------------------------
    auto t_speciate_start = Clock::now();
    speciate();
    adjust_fitness();
    double time_speciate_ms = Ms(Clock::now() - t_speciate_start).count();

    // --- Reproduction / mutation phase --------------------------------------
    auto t_reproduce_start = Clock::now();
    reproduce();
    double time_reproduce_ms = Ms(Clock::now() - t_reproduce_start).count();

    ++generation_;

    double time_total_ms = Ms(Clock::now() - t_total_start).count();

    return GenerationResult{
        evaluated_generation,
        best_fitness,
        mean_fitness,
        worst_fitness,
        num_species(),
        time_eval_ms,
        time_speciate_ms,
        time_reproduce_ms,
        time_total_ms,
    };
}

RunResult Population::run_until(const EvalFn& eval_fn, const StopFn& stop_fn) {
    RunResult run;
    bool stop = false;
    do {
        run.generations.push_back(run_generation(eval_fn));
        stop = stop_fn(run.generations.back());
    } while (!stop);
    run.converged = stop;
    return run;
}

// ---------------------------------------------------------------------------
// Visualization helpers
// ---------------------------------------------------------------------------

static Network build_or_throw(const std::optional<Genome>& snapshot,
                               const Config& cfg,
                               const char* name)
{
    if (!snapshot.has_value()) {
        throw std::logic_error(
            std::string(name) + " called before any generation has run");
    }
    return Network(*snapshot, cfg);
}

Network Population::best_network() const {
    return build_or_throw(snapshot_best_, cfg_, "best_network()");
}

Network Population::worst_network() const {
    return build_or_throw(snapshot_worst_, cfg_, "worst_network()");
}

Network Population::random_network() {
    return build_or_throw(snapshot_random_, cfg_, "random_network()");
}

// ===========================================================================
// Speciation
//
// Each genome is compared against the representative of each existing species.
// If it falls within compat_threshold it joins that species, otherwise a new
// species is created. Representatives are chosen randomly from the previous
// generation's members (Stanley's approach).
// ===========================================================================

void Population::speciate() {
    // Clear members but keep representatives and stagnation state
    for (auto& s : species_) {
        s.members.clear();
    }

    for (auto& genome : genomes_) {
        bool placed = false;
        for (auto& s : species_) {
            double dist = Genome::compatibility_distance(genome, s.representative, cfg_);
            if (dist < cfg_.compat_threshold) {
                s.members.push_back(std::move(genome));
                placed = true;
                break;
            }
        }
        if (!placed) {
            Species ns;
            ns.representative = genome;
            ns.members.push_back(std::move(genome));
            species_.push_back(std::move(ns));
        }
    }

    // Remove extinct species
    species_.erase(
        std::remove_if(species_.begin(), species_.end(),
            [](const Species& s) { return s.members.empty(); }),
        species_.end()
    );

    // Update representatives: pick a random member from the current generation
    for (auto& s : species_) {
        int idx = rng_.random_int(0, static_cast<int>(s.members.size()) - 1);
        s.representative = s.members[idx];
    }
}

// ===========================================================================
// Adjusted fitness
//
// Each genome's adjusted fitness is its raw fitness divided by species size.
// This gives smaller species a proportional share of reproduction, preventing
// any one species from dominating by sheer numbers.
// ===========================================================================

void Population::adjust_fitness() {
    for (auto& s : species_) {
        double best_in_species = 0.0;
        double size = static_cast<double>(s.members.size());

        for (auto& g : s.members) {
            g.adjusted_fitness = g.fitness / size;
            if (g.fitness > best_in_species) {
                best_in_species = g.fitness;
            }
        }

        if (best_in_species > s.best_fitness) {
            s.best_fitness = best_in_species;
            s.stagnation   = 0;
        } else {
            ++s.stagnation;
        }
    }
}

// ===========================================================================
// Reproduction
//
// Offspring are allocated proportionally to each species' share of the total
// adjusted fitness. Stagnant species are culled before allocation (the species
// containing the global best genome is always protected).
//
// Within each species:
//   1. Sort by fitness descending
//   2. Keep top survival_threshold fraction as parents
//   3. Produce offspring via crossover + mutation or clone + mutation
// ===========================================================================

size_t Population::best_species_index() const {
    size_t best_idx  = 0;
    double best_fit  = -1e300;
    for (size_t i = 0; i < species_.size(); ++i) {
        for (const auto& g : species_[i].members) {
            if (g.fitness > best_fit) {
                best_fit = g.fitness;
                best_idx = i;
            }
        }
    }
    return best_idx;
}

void Population::reproduce() {
    size_t protected_idx = best_species_index();

    // Cull stagnant species, protecting the one with the global best genome
    for (size_t i = 0; i < species_.size(); ) {
        if (i != protected_idx && species_[i].stagnation >= cfg_.dropoff_age) {
            species_.erase(species_.begin() + static_cast<long>(i));
            if (protected_idx > i) --protected_idx;
        } else {
            ++i;
        }
    }

    // Compute total adjusted fitness for proportional allocation
    double total_adj = 0.0;
    for (const auto& s : species_) {
        for (const auto& g : s.members) {
            total_adj += g.adjusted_fitness;
        }
    }

    // Assign offspring counts proportionally using largest-remainder method.
    // No per-species minimum — a species with zero fitness gets zero offspring
    // this generation. The remainder pass guarantees the total is exactly
    // population_size.
    std::vector<uint32_t> offspring_counts(species_.size(), 0);
    uint32_t assigned = 0;
    std::vector<double> remainders(species_.size(), 0.0);

    if (total_adj > 0.0) {
        for (size_t i = 0; i < species_.size(); ++i) {
            double species_adj = 0.0;
            for (const auto& g : species_[i].members) {
                species_adj += g.adjusted_fitness;
            }
            double share        = (species_adj / total_adj) * cfg_.population_size;
            offspring_counts[i] = static_cast<uint32_t>(share); // floor
            remainders[i]       = share - std::floor(share);
            assigned           += offspring_counts[i];
        }
    } else {
        // All fitnesses zero: distribute equally across species
        uint32_t equal_share = cfg_.population_size / static_cast<uint32_t>(species_.size());
        for (size_t i = 0; i < species_.size(); ++i) {
            offspring_counts[i] = equal_share;
            remainders[i]       = 1.0; // equal remainder priority
            assigned           += equal_share;
        }
    }

    // Distribute any remaining slots to species with the largest remainders
    while (assigned < cfg_.population_size) {
        size_t best = std::max_element(remainders.begin(), remainders.end())
                      - remainders.begin();
        ++offspring_counts[best];
        remainders[best] = 0.0;
        ++assigned;
    }

    // Reset innovation cache for this generation
    innovations_.next_generation();

    // Breed offspring
    std::vector<Genome> next_generation;
    next_generation.reserve(cfg_.population_size);

    for (size_t si = 0; si < species_.size(); ++si) {
        auto& members = species_[si].members;

        // Sort descending by fitness
        std::sort(members.begin(), members.end(),
            [](const Genome& a, const Genome& b) { return a.fitness > b.fitness; });

        // Cull to top survival_threshold
        size_t survivors = std::max(size_t{1},
            static_cast<size_t>(std::ceil(members.size() * cfg_.survival_threshold)));
        members.resize(survivors);

        uint32_t count = offspring_counts[si];
        for (uint32_t i = 0; i < count; ++i) {
            Genome child = [&]() -> Genome {
                if (members.size() > 1 && rng_.prob(cfg_.prob_crossover)) {
                    int idx_a = rng_.random_int(0, static_cast<int>(members.size()) - 1);
                    int idx_b = rng_.random_int(0, static_cast<int>(members.size()) - 1);
                    // Ensure more_fit is passed as first argument
                    const Genome& a = members[idx_a];
                    const Genome& b = members[idx_b];
                    if (a.fitness >= b.fitness) {
                        return Genome::crossover(a, b, cfg_, rng_);
                    } else {
                        return Genome::crossover(b, a, cfg_, rng_);
                    }
                }
                // Clone a random survivor
                return members[rng_.random_int(0, static_cast<int>(members.size()) - 1)];
            }();

            child.mutate(cfg_, rng_, innovations_);
            next_generation.push_back(std::move(child));
        }
    }

    genomes_ = std::move(next_generation);
}

} // namespace neat
