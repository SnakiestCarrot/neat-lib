#pragma once

#include <cstdint>
#include <random>

namespace neat {

/**
 * @brief A strictly deterministic random number generator.
 * Wraps the Mersenne Twister but implements custom distribution logic
 * to guarantee bit-for-bit reproducibility across different compilers (GCC/Clang/MSVC).
 */
class Random {
public:
    /**
     * @brief Initialize the PRNG with a specific seed.
     */
    explicit Random(uint64_t seed);

    /**
     * @brief Generates a random double in the range [0.0, 1.0).
     * Guaranteed to be identical across all platforms for the same seed.
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
    // The core 64-bit Mersenne Twister engine
    std::mt19937_64 engine;
};

} // namespace neat