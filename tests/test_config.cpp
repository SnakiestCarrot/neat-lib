#include <gtest/gtest.h>
#include <stdexcept>
#include "neat/config.hpp"

// Test that the default configuration is mathematically valid
TEST(ConfigTest, DefaultConfigIsValid) {
    neat::Config config;
    EXPECT_NO_THROW(config.validate());
}

// Test that a population of zero is caught
TEST(ConfigTest, ZeroPopulationThrows) {
    neat::Config config;
    config.population_size = 0;
    EXPECT_THROW(config.validate(), std::invalid_argument);
}

// Test that probabilities over 1.0 are caught
TEST(ConfigTest, InvalidProbabilityThrows) {
    neat::Config config;
    config.prob_mutate_weight = 1.5;
    EXPECT_THROW(config.validate(), std::invalid_argument);
}