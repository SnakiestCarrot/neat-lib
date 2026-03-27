#include <gtest/gtest.h>
#include "neat/random.hpp"

// Test that two PRNGs with the same seed produce the exact same sequence
TEST(RandomTest, StrictDeterminism) {
    uint64_t seed = 999;
    
    neat::Random rng1(seed);
    neat::Random rng2(seed);
    
    for (int i = 0; i < 1000; ++i) {
        EXPECT_DOUBLE_EQ(rng1.random_double(), rng2.random_double());
        EXPECT_EQ(rng1.random_int(1, 100), rng2.random_int(1, 100));
    }
}

// Test that two PRNGs with different seeds diverge immediately
TEST(RandomTest, SeedDivergence) {
    neat::Random rng1(100);
    neat::Random rng2(101);
    
    bool diverged = false;
    for (int i = 0; i < 10; ++i) {
        if (rng1.random_double() != rng2.random_double()) {
            diverged = true;
            break;
        }
    }
    
    EXPECT_TRUE(diverged) << "Different seeds produced the exact same sequence!";
}