#include "neat/random.hpp"

namespace neat {

Random::Random(uint64_t seed) : engine(seed) {}

double Random::random_double() {
    // Manually scale the 64-bit integer to a double in [0.0, 1.0)
    // This avoids implementation-defined behavior in std::uniform_real_distribution
    constexpr double max_val = static_cast<double>(std::mt19937_64::max());
    constexpr double min_val = static_cast<double>(std::mt19937_64::min());
    
    double raw = static_cast<double>(engine());
    return (raw - min_val) / ((max_val - min_val) + 1.0);
}

int Random::random_int(int min, int max) {
    if (min > max) {
        return min; // Fallback for safety, or you could throw an exception
    }
    
    // Manual modulo distribution. 
    // Note: This introduces a tiny bit of modulo bias, but for NEAT topology 
    // mutations, speed and cross-compiler determinism are much more important 
    // than cryptographic uniformity.
    uint64_t range = static_cast<uint64_t>(max - min + 1);
    return min + static_cast<int>(engine() % range);
}

bool Random::prob(double p) {
    if (p <= 0.0) return false;
    if (p >= 1.0) return true;
    return random_double() < p;
}

} // namespace neat