#pragma once

#include <cstdint>
#include <unordered_set>
#include <vector>

namespace neat {

// Forward declarations
struct Config;
class Random;

// ---------------------------------------------------------------------------
// Gene Types — kept as plain structs for data-oriented, cache-friendly layout
// ---------------------------------------------------------------------------

enum class NodeType : uint8_t { INPUT, BIAS, HIDDEN, OUTPUT };

struct NodeGene {
    uint32_t id;
    NodeType type;
};

struct ConnectionGene {
    uint32_t innovation;
    uint32_t from;
    uint32_t to;
    double weight;
    bool enabled;
};

// ---------------------------------------------------------------------------
// Genome — the fundamental unit of evolution in NEAT
// ---------------------------------------------------------------------------

class Genome {
public:
    // --- Construction -------------------------------------------------------

    /// Creates a minimal-topology genome (all inputs connected to all outputs).
    /// This is the starting point for every individual in generation 0.
    static Genome create_minimal(
        uint32_t num_inputs,
        uint32_t num_outputs,
        Random& rng,
        uint32_t& innovation_counter
    );

    // --- Mutation (modifies this genome in-place) ---------------------------

    void mutate(const Config& cfg, Random& rng, uint32_t& innovation_counter);

    void mutate_weights(const Config& cfg, Random& rng);
    void mutate_add_node(Random& rng, uint32_t& innovation_counter);
    void mutate_add_connection(const Config& cfg, Random& rng, uint32_t& innovation_counter);
    void mutate_toggle_enable(Random& rng);

    // --- Crossover ----------------------------------------------------------

    /// Produces a child genome from two parents.
    /// `more_fit` is the fitter parent; `less_fit` is the other.
    /// Disjoint/excess genes are inherited from the more fit parent.
    static Genome crossover(
        const Genome& more_fit,
        const Genome& less_fit,
        const Config& cfg,
        Random& rng
    );

    // --- Speciation ---------------------------------------------------------

    /// Computes the compatibility distance between two genomes using the
    /// formula: d = (c1*E + c2*D) / N + c3*W
    static double compatibility_distance(
        const Genome& a,
        const Genome& b,
        const Config& cfg
    );

    // --- Fitness ------------------------------------------------------------

    double fitness = 0.0;
    double adjusted_fitness = 0.0;

    // --- Node lookup (binary search, nodes kept sorted by id) ----------------

    /// Finds a node by ID using binary search. Returns nullptr if not found.
    const NodeGene* find_node(uint32_t id) const;

    // --- Direct access to gene vectors (data-oriented) ----------------------

    /// Nodes are kept sorted by id. Use find_node() for lookup.
    std::vector<NodeGene> nodes;

    /// Connections are kept sorted by innovation number.
    std::vector<ConnectionGene> connections;

    uint32_t num_inputs() const { return num_inputs_; }
    uint32_t num_outputs() const { return num_outputs_; }

private:
    uint32_t num_inputs_ = 0;
    uint32_t num_outputs_ = 0;

    /// O(1) lookup for existing connections by (from, to) pair.
    /// Key: (from << 32) | to — packs both IDs into a single uint64_t.
    std::unordered_set<uint64_t> connection_set_;

    /// Adds a connection and updates the index. Maintains sorted order.
    void add_connection(const ConnectionGene& conn);

    /// Returns true if a connection from -> to already exists.
    bool has_connection(uint32_t from, uint32_t to) const;

    /// Returns true if adding from -> to would create a cycle.
    bool would_create_cycle(uint32_t from, uint32_t to) const;
};

} // namespace neat
