#include <gtest/gtest.h>
#include "neat/random.hpp"

// ----------------------------------------------------------------------------
// Determinism Tests
// ----------------------------------------------------------------------------

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

// Many naive PRNG algorithms collapse into an infinite loop of zeros if seeded with 0.
// This proves our SplitMix64 initialization successfully prevents this problem.
TEST(RandomTest, SeedZeroWorks) {
    neat::Random rng(0);
    
    bool found_non_zero = false;
    for (int i = 0; i < 100; ++i) {
        if (rng.random_double() > 0.0) {
            found_non_zero = true;
            break;
        }
    }
    EXPECT_TRUE(found_non_zero) << "PRNG produced only zeros when given seed 0!";
}

// Test that random_double() strictly obeys the [0.0, 1.0) boundary
TEST(RandomTest, DoubleBounds) {
    neat::Random rng(42);
    for (int i = 0; i < 10000; ++i) {
        double val = rng.random_double();
        EXPECT_GE(val, 0.0);
        EXPECT_LT(val, 1.0); // Must be strictly less than 1.0
    }
}

// Test edge cases for integer generation
TEST(RandomTest, IntEdgeCases) {
    neat::Random rng(1337);

    EXPECT_EQ(rng.random_int(5, 5), 5);

    for (int i = 0; i < 100; ++i) {
        int neg_val = rng.random_int(-20, -10);
        EXPECT_GE(neg_val, -20);
        EXPECT_LE(neg_val, -10);
    }

    EXPECT_EQ(rng.random_int(10, 5), 10);
}

TEST(RandomTest, ProbEdgeCases) {
    neat::Random rng(123);

    EXPECT_FALSE(rng.prob(0.0));
    EXPECT_FALSE(rng.prob(-1.5));

    EXPECT_TRUE(rng.prob(1.0));
    EXPECT_TRUE(rng.prob(2.5));
}

// A quick check to ensure the distribution isn't horribly skewed.
// Over 100,000 iterations at 50% probability, we expect ~50,000 true results.
TEST(RandomTest, ProbStatisticalSanity) {
    neat::Random rng(42);
    int true_count = 0;
    int iterations = 100000;
    
    for (int i = 0; i < iterations; ++i) {
        if (rng.prob(0.5)) {
            true_count++;
        }
    }
    
    // We give it a generous +/- 1000 buffer to prevent "flaky" tests that fail randomly.
    EXPECT_NEAR(true_count, 50000, 1000);
}