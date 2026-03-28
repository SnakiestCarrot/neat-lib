#pragma once

#include <cstdint>
#include <unordered_map>

namespace neat {

// Tracks structural innovation numbers across an evolutionary run.
//
// Within a single generation, if two separate mutations split the same
// connection (same from->to pair), they must receive the *same* innovation
// number. This is required for correct matching during crossover and accurate
// compatibility distance computation.
//
// Call next_generation() at the start of each new generation to clear the
// within-generation cache while preserving the global counter.
class InnovationTracker {
public:
    explicit InnovationTracker(uint32_t start_counter = 0);

    // Returns the innovation number for a (from, to) structural mutation.
    // If this edge was already mutated this generation, returns the same
    // innovation number assigned then. Otherwise assigns a new one.
    uint32_t get_or_assign(uint32_t from, uint32_t to);

    // Clears the within-generation cache. Call once per generation before
    // mutations are applied.
    void next_generation();

    uint32_t current_counter() const;

private:
    uint32_t counter_;
    std::unordered_map<uint64_t, uint32_t> generation_cache_;
};

} // namespace neat
