#include <gtest/gtest.h>
#include "neat/innovation.hpp"

// ----------------------------------------------------------------------------
// Basic assignment
// ----------------------------------------------------------------------------

TEST(InnovationTrackerTest, FirstAssignmentIncrementsCounter) {
    neat::InnovationTracker tracker;
    EXPECT_EQ(tracker.current_counter(), 0u);

    uint32_t innov = tracker.get_or_assign(0, 1);
    EXPECT_EQ(innov, 0u);
    EXPECT_EQ(tracker.current_counter(), 1u);
}

TEST(InnovationTrackerTest, DifferentEdgesGetDifferentInnovations) {
    neat::InnovationTracker tracker;
    uint32_t a = tracker.get_or_assign(0, 1);
    uint32_t b = tracker.get_or_assign(1, 2);
    uint32_t c = tracker.get_or_assign(0, 2);

    EXPECT_NE(a, b);
    EXPECT_NE(b, c);
    EXPECT_NE(a, c);
    EXPECT_EQ(tracker.current_counter(), 3u);
}

// ----------------------------------------------------------------------------
// Within-generation deduplication — the core NEAT requirement
// ----------------------------------------------------------------------------

TEST(InnovationTrackerTest, SameEdgeSameGenerationGetsSameInnovation) {
    neat::InnovationTracker tracker;

    uint32_t first  = tracker.get_or_assign(1, 3);
    uint32_t second = tracker.get_or_assign(1, 3);

    EXPECT_EQ(first, second);
    // Counter should only have advanced once
    EXPECT_EQ(tracker.current_counter(), 1u);
}

// ----------------------------------------------------------------------------
// next_generation clears the cache but preserves the counter
// ----------------------------------------------------------------------------

TEST(InnovationTrackerTest, NextGenerationClearsCache) {
    neat::InnovationTracker tracker;

    uint32_t gen1 = tracker.get_or_assign(1, 3);
    EXPECT_EQ(tracker.current_counter(), 1u);

    tracker.next_generation();

    // Same edge in a new generation gets a new, higher innovation number
    uint32_t gen2 = tracker.get_or_assign(1, 3);
    EXPECT_NE(gen1, gen2);
    EXPECT_EQ(tracker.current_counter(), 2u);
}

TEST(InnovationTrackerTest, NextGenerationPreservesCounter) {
    neat::InnovationTracker tracker;
    tracker.get_or_assign(0, 1);
    tracker.get_or_assign(1, 2);
    EXPECT_EQ(tracker.current_counter(), 2u);

    tracker.next_generation();
    EXPECT_EQ(tracker.current_counter(), 2u); // counter unchanged after clear
}

// ----------------------------------------------------------------------------
// Custom start counter
// ----------------------------------------------------------------------------

TEST(InnovationTrackerTest, CustomStartCounter) {
    neat::InnovationTracker tracker(100);
    EXPECT_EQ(tracker.current_counter(), 100u);

    uint32_t innov = tracker.get_or_assign(0, 1);
    EXPECT_EQ(innov, 100u);
    EXPECT_EQ(tracker.current_counter(), 101u);
}
