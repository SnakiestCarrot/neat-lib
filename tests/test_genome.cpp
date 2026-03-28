#include <gtest/gtest.h>
#include "neat/genome.hpp"
#include "neat/innovation.hpp"
#include "neat/config.hpp"
#include "neat/random.hpp"

#include <algorithm>
#include <unordered_set>

// Helper: create a standard 2-input, 1-output minimal genome
static neat::Genome make_genome(neat::InnovationTracker& innovations) {
    neat::Random rng(42);
    return neat::Genome::create_minimal(2, 1, rng, innovations);
}

// ============================================================================
// create_minimal structure
// ============================================================================

TEST(GenomeTest, MinimalNodeLayout) {
    neat::InnovationTracker innovations;
    neat::Genome g = make_genome(innovations);

    // 1 bias + 2 inputs + 1 output = 4 nodes
    ASSERT_EQ(g.nodes.size(), 4u);
    EXPECT_EQ(g.nodes[0].type, neat::NodeType::BIAS);
    EXPECT_EQ(g.nodes[1].type, neat::NodeType::INPUT);
    EXPECT_EQ(g.nodes[2].type, neat::NodeType::INPUT);
    EXPECT_EQ(g.nodes[3].type, neat::NodeType::OUTPUT);
}

TEST(GenomeTest, MinimalConnectionCount) {
    neat::InnovationTracker innovations;
    neat::Genome g = make_genome(innovations);

    // (2 inputs + 1 bias) * 1 output = 3 connections
    EXPECT_EQ(g.connections.size(), 3u);
    for (const auto& c : g.connections) {
        EXPECT_TRUE(c.enabled);
    }
}

TEST(GenomeTest, MinimalConnectionsSortedByInnovation) {
    neat::InnovationTracker innovations;
    neat::Genome g = make_genome(innovations);

    for (size_t i = 1; i < g.connections.size(); ++i) {
        EXPECT_LT(g.connections[i-1].innovation, g.connections[i].innovation);
    }
}

TEST(GenomeTest, MinimalNumInputsOutputs) {
    neat::InnovationTracker innovations;
    neat::Genome g = make_genome(innovations);
    EXPECT_EQ(g.num_inputs(), 2u);
    EXPECT_EQ(g.num_outputs(), 1u);
}

// ============================================================================
// find_node
// ============================================================================

TEST(GenomeTest, FindNodeReturnsCorrectNode) {
    neat::InnovationTracker innovations;
    neat::Genome g = make_genome(innovations);

    const neat::NodeGene* bias = g.find_node(0);
    ASSERT_NE(bias, nullptr);
    EXPECT_EQ(bias->type, neat::NodeType::BIAS);

    const neat::NodeGene* out = g.find_node(3);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->type, neat::NodeType::OUTPUT);
}

TEST(GenomeTest, FindNodeReturnsNullForMissingId) {
    neat::InnovationTracker innovations;
    neat::Genome g = make_genome(innovations);
    EXPECT_EQ(g.find_node(99), nullptr);
}

// ============================================================================
// mutate_add_node
// ============================================================================

TEST(GenomeTest, AddNodeIncreasesNodeAndConnectionCount) {
    neat::InnovationTracker innovations;
    neat::Random rng(1);
    neat::Genome g = make_genome(innovations);

    size_t nodes_before = g.nodes.size();
    size_t conns_before = g.connections.size();

    g.mutate_add_node(rng, innovations);

    EXPECT_EQ(g.nodes.size(), nodes_before + 1);
    // One connection disabled, two new ones added
    EXPECT_EQ(g.connections.size(), conns_before + 2);
}

TEST(GenomeTest, AddNodeDisablesOriginalConnection) {
    neat::InnovationTracker innovations;
    neat::Random rng(1);
    neat::Genome g = make_genome(innovations);

    g.mutate_add_node(rng, innovations);

    // Exactly one connection should be disabled
    int disabled = 0;
    for (const auto& c : g.connections) {
        if (!c.enabled) ++disabled;
    }
    EXPECT_EQ(disabled, 1);
}

TEST(GenomeTest, AddNodeNewConnectionWeights) {
    neat::InnovationTracker innovations;
    neat::Random rng(1);
    neat::Genome g = make_genome(innovations);

    // Record which connection gets split
    // Find first enabled connection to determine expected preserved weight
    double split_weight = -1.0;
    for (const auto& c : g.connections) {
        if (c.enabled) { split_weight = c.weight; break; }
    }

    g.mutate_add_node(rng, innovations);

    // The two new connections are the last two added (highest innovation numbers)
    // source->hidden has weight 1.0, hidden->target preserves old weight
    const auto& last = g.connections.back();
    const auto& second_last = g.connections[g.connections.size() - 2];

    // One of the two new connections must have weight 1.0
    bool one_is_identity = (last.weight == 1.0 || second_last.weight == 1.0);
    EXPECT_TRUE(one_is_identity);
}

// ============================================================================
// mutate_add_connection
// ============================================================================

TEST(GenomeTest, AddConnectionIncreasesConnectionCount) {
    neat::InnovationTracker innovations;
    neat::Config cfg;
    neat::Random rng(42);

    // Use a genome with a hidden node so there are valid new connections to add
    neat::Genome g = make_genome(innovations);
    g.mutate_add_node(rng, innovations);

    size_t before = g.connections.size();
    g.mutate_add_connection(cfg, rng, innovations);

    EXPECT_GT(g.connections.size(), before);
}

TEST(GenomeTest, AddConnectionNoDuplicates) {
    neat::InnovationTracker innovations;
    neat::Config cfg;
    neat::Random rng(42);
    neat::Genome g = make_genome(innovations);
    g.mutate_add_node(rng, innovations);

    // Run many times and verify no duplicate (from, to) pairs
    for (int i = 0; i < 20; ++i) {
        g.mutate_add_connection(cfg, rng, innovations);
    }

    std::unordered_set<uint64_t> seen;
    for (const auto& c : g.connections) {
        uint64_t key = (static_cast<uint64_t>(c.from) << 32) | c.to;
        EXPECT_TRUE(seen.insert(key).second) << "Duplicate connection from " << c.from << " to " << c.to;
    }
}

// ============================================================================
// Innovation deduplication across genomes in the same generation
// ============================================================================

TEST(GenomeTest, SameStructuralMutationSameGenerationSharesInnovation) {
    neat::InnovationTracker innovations;
    neat::Random rng1(10);
    neat::Random rng2(20);

    neat::Genome g1 = neat::Genome::create_minimal(2, 1, rng1, innovations);
    neat::Genome g2 = neat::Genome::create_minimal(2, 1, rng2, innovations);

    // Both genomes split the same connection (bias->output, innovation 0)
    // by disabling it manually and calling add_node on the same edge.
    // We simulate this by directly calling mutate_add_node with the same rng
    // seed so they pick the same connection to split.
    neat::Random rng_a(99);
    neat::Random rng_b(99); // same seed -> same connection chosen

    g1.mutate_add_node(rng_a, innovations);
    g2.mutate_add_node(rng_b, innovations);

    // The two new connections in g1 and g2 should share innovation numbers
    // since they split the same edge in the same generation.
    uint32_t g1_innov_last       = g1.connections.back().innovation;
    uint32_t g1_innov_second     = g1.connections[g1.connections.size()-2].innovation;
    uint32_t g2_innov_last       = g2.connections.back().innovation;
    uint32_t g2_innov_second     = g2.connections[g2.connections.size()-2].innovation;

    EXPECT_EQ(g1_innov_last,   g2_innov_last);
    EXPECT_EQ(g1_innov_second, g2_innov_second);
}

// ============================================================================
// Crossover
// ============================================================================

TEST(GenomeTest, CrossoverChildHasCorrectInputsOutputs) {
    neat::InnovationTracker innovations;
    neat::Config cfg;
    neat::Random rng(7);

    neat::Genome parent_a = make_genome(innovations);
    neat::Genome parent_b = make_genome(innovations);

    neat::Genome child = neat::Genome::crossover(parent_a, parent_b, cfg, rng);
    EXPECT_EQ(child.num_inputs(),  2u);
    EXPECT_EQ(child.num_outputs(), 1u);
}

TEST(GenomeTest, CrossoverChildConnectionsSortedByInnovation) {
    neat::InnovationTracker innovations;
    neat::Config cfg;
    neat::Random rng(7);

    neat::Genome parent_a = make_genome(innovations);
    neat::Genome parent_b = make_genome(innovations);
    parent_a.mutate_add_node(rng, innovations);

    neat::Genome child = neat::Genome::crossover(parent_a, parent_b, cfg, rng);
    for (size_t i = 1; i < child.connections.size(); ++i) {
        EXPECT_LT(child.connections[i-1].innovation, child.connections[i].innovation);
    }
}

TEST(GenomeTest, CrossoverDisjointExcessFromMoreFitParent) {
    neat::InnovationTracker innovations;
    neat::Config cfg;
    neat::Random rng(7);

    // more_fit has extra connections that less_fit does not
    neat::Genome more_fit = make_genome(innovations);
    neat::Genome less_fit = make_genome(innovations);
    more_fit.mutate_add_node(rng, innovations);

    neat::Genome child = neat::Genome::crossover(more_fit, less_fit, cfg, rng);

    // Child must have at least as many connections as more_fit
    EXPECT_GE(child.connections.size(), more_fit.connections.size());
}

// ============================================================================
// Compatibility distance
// ============================================================================

TEST(GenomeTest, CompatibilityDistanceIdenticalGenomesIsZero) {
    neat::InnovationTracker innovations;
    neat::Config cfg;
    neat::Genome g = make_genome(innovations);

    double dist = neat::Genome::compatibility_distance(g, g, cfg);
    EXPECT_DOUBLE_EQ(dist, 0.0);
}

TEST(GenomeTest, CompatibilityDistanceSymmetric) {
    neat::InnovationTracker innovations;
    neat::Config cfg;
    neat::Random rng(1);

    neat::Genome a = make_genome(innovations);
    neat::Genome b = make_genome(innovations);
    b.mutate_add_node(rng, innovations);

    double ab = neat::Genome::compatibility_distance(a, b, cfg);
    double ba = neat::Genome::compatibility_distance(b, a, cfg);
    EXPECT_DOUBLE_EQ(ab, ba);
}

TEST(GenomeTest, CompatibilityDistanceIncreasesWithStructuralDifference) {
    neat::InnovationTracker innovations;
    neat::Config cfg;
    neat::Random rng(1);

    neat::Genome base  = make_genome(innovations);
    neat::Genome small_diff = make_genome(innovations);
    neat::Genome large_diff = make_genome(innovations);

    small_diff.mutate_add_node(rng, innovations);
    large_diff.mutate_add_node(rng, innovations);
    large_diff.mutate_add_node(rng, innovations);
    large_diff.mutate_add_node(rng, innovations);

    double d_small = neat::Genome::compatibility_distance(base, small_diff, cfg);
    double d_large = neat::Genome::compatibility_distance(base, large_diff, cfg);

    EXPECT_LT(d_small, d_large);
}
