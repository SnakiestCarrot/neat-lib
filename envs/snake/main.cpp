#include <iostream>
#include <iomanip>
#include <stdexcept>

// Include our new library headers!
#include "neat/config.hpp"
#include "neat/random.hpp"

int main() {
    std::cout << "--- NEAT Engine Test ---\n";

    try {
        // 1. Test Configuration and Validation
        neat::Config config {
            .seed = 1337,                 // Set a specific seed for testing
            .population_size = 150,
            .prob_mutate_weight = 0.85    // Tweak a default value
        };

        std::cout << "[Config] Validating parameters...\n";
        config.validate();
        std::cout << "[Config] Validation passed! Seed: " << config.seed << "\n\n";

        // 2. Test the PRNG Determinism
        neat::Random rng(config.seed);

        std::cout << "[PRNG] Generating 5 random doubles [0.0, 1.0):\n";
        for (int i = 0; i < 5; ++i) {
            std::cout << "  " << std::fixed << std::setprecision(6) << rng.random_double() << "\n";
        }

        std::cout << "\n[PRNG] Generating 5 random integers [1, 100]:\n";
        for (int i = 0; i < 5; ++i) {
            std::cout << "  " << rng.random_int(1, 100) << "\n";
        }

        std::cout << "\n[PRNG] Simulating 5 weight mutations (85% chance):\n";
        for (int i = 0; i < 5; ++i) {
            bool mutate = rng.prob(config.prob_mutate_weight);
            std::cout << "  Mutation " << i + 1 << ": " << (mutate ? "YES" : "NO") << "\n";
        }

    } catch (const std::exception& e) {
        // If we messed up our config (e.g., prob_mutate_weight = 1.5), it will catch here.
        std::cerr << "\n[ERROR] " << e.what() << "\n";
        return 1;
    }

    std::cout << "\n--- Test Complete ---\n";
    return 0;
}