#include <gtest/gtest.h>
#include "neat/genome.hpp"
#include "neat/network.hpp"
#include "neat/innovation.hpp"
#include "neat/config.hpp"
#include "neat/random.hpp"

#include <cmath>

// Sigmoid helper to hand-compute expected values
static double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

// Builds a minimal 2-input, 1-output genome with known weights, bypassing
// mutation so the structure is fully deterministic.
//
// Node layout (as created by Genome::create_minimal):
//   id 0 = BIAS
//   id 1 = INPUT
//   id 2 = INPUT
//   id 3 = OUTPUT
//
// Connections (manually replaced after create_minimal):
//   bias(0)  -> output(3)  weight = 0.5
//   input(1) -> output(3)  weight = 1.0
//   input(2) -> output(3)  weight = -1.0
static neat::Genome make_simple_genome() {
    neat::Random rng(0);
    neat::InnovationTracker innovations;
    neat::Genome g = neat::Genome::create_minimal(2, 1, rng, innovations);

    // Overwrite weights with known values for deterministic testing
    // Connection order in a minimal 2-input, 1-output genome:
    //   0: bias   -> output
    //   1: input1 -> output
    //   2: input2 -> output
    g.connections[0].weight =  0.5;
    g.connections[1].weight =  1.0;
    g.connections[2].weight = -1.0;

    return g;
}

// ----------------------------------------------------------------------------
// Basic activation tests
// ----------------------------------------------------------------------------

TEST(NetworkTest, MinimalGenomeCorrectOutput) {
    neat::Config cfg;
    cfg.activation = neat::ActivationType::SIGMOID;

    neat::Network net(make_simple_genome(), cfg);

    // inputs = {0.5, 0.5}
    // sum = bias*0.5 + 0.5*1.0 + 0.5*(-1.0)
    //     = 1.0*0.5 + 0.5 - 0.5 = 0.5
    // output = sigmoid(0.5)
    auto out = net.activate({0.5, 0.5});
    ASSERT_EQ(out.size(), 1u);
    EXPECT_DOUBLE_EQ(out[0], sigmoid(0.5));
}

TEST(NetworkTest, MinimalGenomeZeroInputs) {
    neat::Config cfg;
    cfg.activation = neat::ActivationType::SIGMOID;

    neat::Network net(make_simple_genome(), cfg);

    // inputs = {0.0, 0.0}
    // sum = 1.0*0.5 + 0.0 + 0.0 = 0.5
    auto out = net.activate({0.0, 0.0});
    ASSERT_EQ(out.size(), 1u);
    EXPECT_DOUBLE_EQ(out[0], sigmoid(0.5));
}

// ----------------------------------------------------------------------------
// Bias contributes 1.0 regardless of inputs
// ----------------------------------------------------------------------------

TEST(NetworkTest, BiasAlwaysOne) {
    neat::Config cfg;
    cfg.activation = neat::ActivationType::SIGMOID;

    // Genome with only a bias->output connection (weight 1.0), no input connections
    neat::Random rng(0);
    neat::InnovationTracker innovations;
    neat::Genome g = neat::Genome::create_minimal(2, 1, rng, innovations);

    // Disable input connections, keep only bias->output
    g.connections[1].enabled = false;
    g.connections[2].enabled = false;
    g.connections[0].weight = 1.0;

    neat::Network net(g, cfg);
    auto out = net.activate({99.0, -99.0});
    ASSERT_EQ(out.size(), 1u);
    // Only bias contributes: sigmoid(1.0 * 1.0)
    EXPECT_DOUBLE_EQ(out[0], sigmoid(1.0));
}

// ----------------------------------------------------------------------------
// Hidden node — tests that topological sort produces the correct order
// ----------------------------------------------------------------------------

TEST(NetworkTest, HiddenNodeCorrectOutput) {
    neat::Config cfg;
    cfg.activation = neat::ActivationType::SIGMOID;

    // Build a minimal 1-input, 1-output genome, then manually add a hidden node
    // by splitting the input->output connection.
    //
    // After split the structure is:
    //   bias(0)   -> output(2)  weight = w_bias_out  (original bias->output)
    //   input(1)  -> hidden(3)  weight = 1.0
    //   hidden(3) -> output(2)  weight = w_orig
    //   input(1)  -> output(2)  weight = w_input_out (original, now disabled)
    //
    // We set weights manually for a hand-computable result.
    neat::Random rng(0);
    neat::InnovationTracker innovations;
    neat::Genome g = neat::Genome::create_minimal(1, 1, rng, innovations);

    // Known weights on the initial minimal genome:
    //   conn[0]: bias(0)  -> output(1)
    //   conn[1]: input(1) -> output(2)  <-- we'll split this one

    // Fix weights so we control the computation
    g.connections[0].weight = 0.0; // bias->output: contributes nothing
    g.connections[1].weight = 2.0; // input->output: will be split

    // Perform add-node mutation on the input->output connection (index 1)
    // Manually replicate what mutate_add_node does so weights are known:
    g.connections[1].enabled = false;
    uint32_t new_id = static_cast<uint32_t>(g.nodes.size()); // id = 3
    g.nodes.push_back({new_id, neat::NodeType::HIDDEN});
    g.connections.push_back({innovations.get_or_assign(1, new_id), 1u, new_id, 1.0, true});
    g.connections.push_back({innovations.get_or_assign(new_id, 2u), new_id, 2u, 2.0, true});

    neat::Network net(g, cfg);

    // input = 1.0
    // hidden = sigmoid(1.0 * 1.0) = sigmoid(1.0)
    // output = sigmoid(sigmoid(1.0) * 2.0 + 0.0)
    double hidden_val = sigmoid(1.0);
    double expected   = sigmoid(hidden_val * 2.0);

    auto out = net.activate({1.0});
    ASSERT_EQ(out.size(), 1u);
    EXPECT_DOUBLE_EQ(out[0], expected);
}

// ----------------------------------------------------------------------------
// Wrong input count throws
// ----------------------------------------------------------------------------

TEST(NetworkTest, WrongInputCountThrows) {
    neat::Config cfg;
    neat::Network net(make_simple_genome(), cfg);

    EXPECT_THROW(net.activate({1.0}),         std::invalid_argument); // too few
    EXPECT_THROW(net.activate({1.0, 2.0, 3.0}), std::invalid_argument); // too many
}

// ----------------------------------------------------------------------------
// Activation function variants
// ----------------------------------------------------------------------------

TEST(NetworkTest, TanhActivation) {
    neat::Config cfg;
    cfg.activation = neat::ActivationType::TANH;

    neat::Network net(make_simple_genome(), cfg);

    // sum = 0.5 (same as MinimalGenomeCorrectOutput)
    auto out = net.activate({0.5, 0.5});
    ASSERT_EQ(out.size(), 1u);
    EXPECT_DOUBLE_EQ(out[0], std::tanh(0.5));
}

TEST(NetworkTest, ReluActivation) {
    neat::Config cfg;
    cfg.activation = neat::ActivationType::RELU;

    neat::Network net(make_simple_genome(), cfg);

    // sum = 0.5 -> relu(0.5) = 0.5
    auto out = net.activate({0.5, 0.5});
    ASSERT_EQ(out.size(), 1u);
    EXPECT_DOUBLE_EQ(out[0], 0.5);

    // Negative sum: input weights cancel bias, sum goes negative
    // sum = 1.0*0.5 + 2.0*1.0 + 2.0*(-1.0) = 0.5 -> still positive
    // Force negative: use inputs that make sum negative
    // sum = 0.5 + 0.0*1.0 + 1.0*(-1.0) = -0.5 -> relu = 0.0
    auto out2 = net.activate({0.0, 1.0});
    ASSERT_EQ(out2.size(), 1u);
    EXPECT_DOUBLE_EQ(out2[0], 0.0);
}
