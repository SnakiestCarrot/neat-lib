#pragma once

#include <cstdint>

namespace neat {

/**
 * @brief A strictly deterministic, high-performance random number generator.
 * Implements Xoshiro256** for blazingly fast generation with a tiny 32-byte 
 * cache footprint, seeded via SplitMix64.
 */
class Random {
public:
    /**
     * @brief Initialize the PRNG with a specific master seed.
     */
    explicit Random(uint64_t seed);

    /**
     * @brief Generates a random double in the range [0.0, 1.0).
     */
    double random_double();

    /**
     * @brief Generates a random integer in the range [min, max].
     */
    int random_int(int min, int max);

    /**
     * @brief Returns true with probability 'p'.
     */
    bool prob(double p);

private:
    // The entire state required for Xoshiro256**. 
    // Fits perfectly into half a CPU cache line.
    uint64_t state[4];

    // Core generation engine
    uint64_t next();
    
    // Bitwise left rotation helper
    static inline uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }
};

} // namespace neat