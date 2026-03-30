#include <gtest/gtest.h>
#include "neat/population.hpp"
#include "neat/config.hpp"

// Helper: build a valid config ready for Population construction
static neat::Config make_config(uint32_t pop_size = 20) {
    neat::Config cfg;
    cfg.num_inputs      = 2;
    cfg.num_outputs     = 1;
    cfg.population_size = pop_size;
    // Disable structural mutations so genomes stay comparable across tests
    cfg.prob_add_node   = 0.0;
    cfg.prob_add_link   = 0.0;
    // Disable parallel eval by default — tests use shared-state lambdas that
    // are not thread-safe. Parallelism is tested explicitly below.
    cfg.parallel_eval   = false;
    return cfg;
}

// Eval function that assigns a fixed fitness of 1.0 to every genome
static auto constant_eval = [](neat::Network&) { return 1.0; };

// Eval function that assigns incrementing fitness values
static auto incrementing_eval = [](neat::Network&) {
    static double f = 0.0;
    return ++f;
};

// ============================================================================
// Construction
// ============================================================================

TEST(PopulationTest, ConstructionPopulationSize) {
    neat::Population pop(make_config(30));
    EXPECT_EQ(pop.genomes().size(), 30u);
}

TEST(PopulationTest, ConstructionGenomeTopology) {
    neat::Population pop(make_config());
    for (const auto& g : pop.genomes()) {
        EXPECT_EQ(g.num_inputs(),  2u);
        EXPECT_EQ(g.num_outputs(), 1u);
    }
}

TEST(PopulationTest, ConstructionGenerationIsZero) {
    neat::Population pop(make_config());
    EXPECT_EQ(pop.generation(), 0u);
}

TEST(PopulationTest, InvalidConfigThrows) {
    neat::Config bad_cfg; // num_inputs and num_outputs both 0
    EXPECT_THROW(neat::Population{bad_cfg}, std::invalid_argument);
}

// ============================================================================
// run_generation
// ============================================================================

TEST(PopulationTest, RunGenerationIncrementsGeneration) {
    neat::Population pop(make_config());
    pop.run_generation(constant_eval);
    EXPECT_EQ(pop.generation(), 1u);
    pop.run_generation(constant_eval);
    EXPECT_EQ(pop.generation(), 2u);
}

TEST(PopulationTest, RunGenerationReturnsCorrectGenerationNumber) {
    neat::Population pop(make_config());
    auto result = pop.run_generation(constant_eval);
    EXPECT_EQ(result.generation, 0u); // reflects the generation that was evaluated
}

TEST(PopulationTest, RunGenerationResultFitnessValues) {
    neat::Population pop(make_config(10));
    double counter = 0.0;
    auto eval = [&](neat::Network&) { return ++counter; };
    auto result = pop.run_generation(eval);

    EXPECT_GT(result.best_fitness,  result.mean_fitness);
    EXPECT_LT(result.worst_fitness, result.mean_fitness);
    EXPECT_GE(result.best_fitness,  result.worst_fitness);
}

TEST(PopulationTest, RunGenerationResultNumSpecies) {
    neat::Population pop(make_config());
    auto result = pop.run_generation(constant_eval);
    EXPECT_EQ(result.num_species, pop.num_species());
}

TEST(PopulationTest, PopulationSizePreservedAfterRunGenerationZeroFitness) {
    neat::Config cfg = make_config(50);
    neat::Population pop(cfg);

    for (int gen = 0; gen < 5; ++gen) {
        pop.run_generation([](neat::Network&) { return 0.0; });
        EXPECT_EQ(pop.genomes().size(), 50u)
            << "Population size changed at generation " << pop.generation();
    }
}

TEST(PopulationTest, PopulationSizePreservedAfterRunGenerationWithFitness) {
    neat::Config cfg = make_config(50);
    neat::Population pop(cfg);

    double f = 0.0;
    for (int gen = 0; gen < 5; ++gen) {
        pop.run_generation([&](neat::Network&) { return ++f; });
        EXPECT_EQ(pop.genomes().size(), 50u)
            << "Population size changed at generation " << pop.generation();
    }
}

// ============================================================================
// run_until
// ============================================================================

TEST(PopulationTest, RunUntilStopsAtCorrectGeneration) {
    neat::Population pop(make_config());
    double f = 0.0;
    auto result = pop.run_until(
        [&](neat::Network&) { return ++f; },
        [](const neat::GenerationResult& r) { return r.generation >= 4; }
    );
    EXPECT_EQ(result.generation, 4u);
    EXPECT_EQ(pop.generation(),  5u); // epoch() was called after the final eval
}

TEST(PopulationTest, RunUntilStopsImmediatelyIfConditionMetFirstGen) {
    neat::Population pop(make_config());
    auto result = pop.run_until(
        constant_eval,
        [](const neat::GenerationResult&) { return true; }
    );
    EXPECT_EQ(result.generation, 0u);
}

// ============================================================================
// Visualization helpers
// ============================================================================

TEST(PopulationTest, NetworkHelpersThrowBeforeAnyGeneration) {
    neat::Population pop(make_config());
    EXPECT_THROW(pop.best_network(),   std::logic_error);
    EXPECT_THROW(pop.worst_network(),  std::logic_error);
    EXPECT_THROW(pop.random_network(), std::logic_error);
}

TEST(PopulationTest, NetworkHelpersReturnAfterRunGeneration) {
    neat::Population pop(make_config());
    pop.run_generation(constant_eval);
    EXPECT_NO_THROW(pop.best_network());
    EXPECT_NO_THROW(pop.worst_network());
    EXPECT_NO_THROW(pop.random_network());
}

TEST(PopulationTest, BestNetworkProducesOutputWithCorrectSize) {
    neat::Population pop(make_config());
    pop.run_generation(constant_eval);
    neat::Network net = pop.best_network();
    auto output = net.activate({1.0, 0.5});
    EXPECT_EQ(output.size(), 1u);
}

TEST(PopulationTest, BestNetworkHasHigherFitnessThanWorst) {
    neat::Population pop(make_config(20));
    double f = 0.0;
    // Assign distinct fitness values so best and worst are clearly different
    auto result = pop.run_generation([&](neat::Network&) { return ++f; });
    EXPECT_GE(result.best_fitness, result.worst_fitness);
}

// ============================================================================
// Speciation
// ============================================================================

TEST(PopulationTest, AllMinimalGenomesInOneSpeciesInitially) {
    neat::Population pop(make_config(30));
    pop.run_generation(constant_eval);
    EXPECT_EQ(pop.num_species(), 1u);
}

TEST(PopulationTest, TightCompatThresholdCreatesManySpecies) {
    neat::Config cfg = make_config(20);
    cfg.compat_threshold = 0.01;
    neat::Population pop(cfg);
    pop.run_generation(constant_eval);
    EXPECT_GT(pop.num_species(), 1u);
}

TEST(PopulationTest, NumSpeciesNeverZeroAfterEpoch) {
    neat::Population pop(make_config());
    for (int gen = 0; gen < 10; ++gen) {
        pop.run_generation(constant_eval);
        EXPECT_GT(pop.num_species(), 0u);
    }
}

// ============================================================================
// Stagnation — best species is always protected
// ============================================================================

TEST(PopulationTest, StagnantSpeciesDoesNotCollapsePopulation) {
    neat::Config cfg = make_config(30);
    cfg.compat_threshold = 0.01;
    cfg.dropoff_age      = 2;
    neat::Population pop(cfg);

    double best_f = 0.0;
    for (int gen = 0; gen < 10; ++gen) {
        // First genome always gets the highest fitness — ensures a protected species
        bool first = true;
        pop.run_generation([&](neat::Network&) -> double {
            if (first) { first = false; return ++best_f * 100.0; }
            return 1.0;
        });

        EXPECT_GT(pop.genomes().size(), 0u);
        EXPECT_GT(pop.num_species(),    0u);
        EXPECT_EQ(pop.genomes().size(), 30u);
    }
}

// ============================================================================
// Determinism
// ============================================================================

TEST(PopulationTest, SameSeedProducesSameResults) {
    neat::Config cfg = make_config(20);

    neat::Population pop1(cfg);
    neat::Population pop2(cfg);

    for (int gen = 0; gen < 5; ++gen) {
        double f = 0.0;
        pop1.run_generation([&](neat::Network&) { return ++f; });
        f = 0.0;
        pop2.run_generation([&](neat::Network&) { return ++f; });
    }

    const auto& g1 = pop1.genomes()[0];
    const auto& g2 = pop2.genomes()[0];
    ASSERT_EQ(g1.connections.size(), g2.connections.size());
    for (size_t i = 0; i < g1.connections.size(); ++i) {
        EXPECT_DOUBLE_EQ(g1.connections[i].weight,     g2.connections[i].weight);
        EXPECT_EQ(g1.connections[i].innovation, g2.connections[i].innovation);
    }
}

// ============================================================================
// Parallelism — determinism guarantees
// ============================================================================

// Helper: run a population for N generations and return the connection weights
// of the first genome afterwards. Uses a stateless eval so it is safe to call
// with parallel_eval=true.
static std::vector<double> run_and_collect(neat::Config cfg, int generations) {
    neat::Population pop(cfg);
    // Fitness is derived purely from network output — no shared mutable state,
    // so this lambda is safe to call concurrently from multiple threads.
    auto eval = [](neat::Network& net) { return net.activate({1.0, 0.5})[0]; };
    for (int i = 0; i < generations; ++i) {
        pop.run_generation(eval);
    }
    std::vector<double> weights;
    for (const auto& c : pop.genomes()[0].connections) {
        weights.push_back(c.weight);
    }
    return weights;
}

TEST(PopulationTest, ParallelAndSequentialProduceSameResults) {
    // Both configs share the same seed. One uses parallel eval, the other
    // sequential. epoch() is always single-threaded so the outcome must match.
    neat::Config seq_cfg = make_config(20);
    seq_cfg.parallel_eval = false;

    neat::Config par_cfg = make_config(20);
    par_cfg.parallel_eval = true;

    auto seq_weights = run_and_collect(seq_cfg, 5);
    auto par_weights = run_and_collect(par_cfg, 5);

    ASSERT_EQ(seq_weights.size(), par_weights.size());
    for (size_t i = 0; i < seq_weights.size(); ++i) {
        EXPECT_DOUBLE_EQ(seq_weights[i], par_weights[i])
            << "Weight mismatch at connection " << i;
    }
}

TEST(PopulationTest, TwoParallelRunsSameSeedProduceSameResults) {
    // Two parallel populations with the same seed must converge identically.
    neat::Config cfg = make_config(20);
    cfg.parallel_eval = true;

    auto weights1 = run_and_collect(cfg, 5);
    auto weights2 = run_and_collect(cfg, 5);

    ASSERT_EQ(weights1.size(), weights2.size());
    for (size_t i = 0; i < weights1.size(); ++i) {
        EXPECT_DOUBLE_EQ(weights1[i], weights2[i])
            << "Weight mismatch at connection " << i;
    }
}

TEST(PopulationTest, DifferentSeedsProduceDifferentResults) {
    neat::Config cfg1 = make_config(20);
    neat::Config cfg2 = make_config(20);
    cfg2.seed = cfg1.seed + 1;

    neat::Population pop1(cfg1);
    neat::Population pop2(cfg2);

    pop1.run_generation(constant_eval);
    pop2.run_generation(constant_eval);

    bool diverged = false;
    const auto& g1 = pop1.genomes()[0];
    const auto& g2 = pop2.genomes()[0];
    for (size_t i = 0; i < std::min(g1.connections.size(), g2.connections.size()); ++i) {
        if (g1.connections[i].weight != g2.connections[i].weight) {
            diverged = true;
            break;
        }
    }
    EXPECT_TRUE(diverged);
}
