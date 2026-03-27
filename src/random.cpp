#include "neat/random.hpp"

namespace neat {

// ----------------------------------------------------------------------------
// Initialization (SplitMix64)
// ----------------------------------------------------------------------------
Random::Random(uint64_t seed) {
    // We use SplitMix64 to expand the 64-bit seed into 256 bits of state.
    // It also guarantees the starting state is never all zeros (which would break Xoshiro).
    auto splitmix64 = [&seed]() -> uint64_t {
        uint64_t z = (seed += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    };

    state[0] = splitmix64();
    state[1] = splitmix64();
    state[2] = splitmix64();
    state[3] = splitmix64();
}

// ----------------------------------------------------------------------------
// The Xoshiro256** Engine
// ----------------------------------------------------------------------------
uint64_t Random::next() {
    const uint64_t result = rotl(state[1] * 5, 7) * 9;
    const uint64_t t = state[1] << 17;

    state[2] ^= state[0];
    state[3] ^= state[1];
    state[1] ^= state[2];
    state[0] ^= state[3];
    state[2] ^= t;
    state[3] = rotl(state[3], 45);

    return result;
}

// ----------------------------------------------------------------------------
// Public Interface
// ----------------------------------------------------------------------------
double Random::random_double() {
    // Standard IEEE 754 doubles have 53 bits of precision.
    // We take the top 53 bits of our generated integer and multiply by 2^-53.
    // This is mathematically the fastest and most deterministic way to get [0.0, 1.0).
    return (next() >> 11) * (1.0 / (1ULL << 53));
}

int Random::random_int(int min, int max) {
    if (min > max) return min;
    
    uint64_t range = static_cast<uint64_t>(max - min + 1);
    return min + static_cast<int>(next() % range);
}

bool Random::prob(double p) {
    if (p <= 0.0) return false;
    if (p >= 1.0) return true;
    return random_double() < p;
}

} // namespace neat