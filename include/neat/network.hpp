#pragma once

#include "neat/genome.hpp"

#include <cstdint>
#include <vector>

namespace neat {

struct Config;

// A phenotype built from a Genome. Activation order is computed once at
// construction via Kahn's topological sort, so forward passes are a simple
// linear scan with no graph traversal at runtime.
class Network {
public:
    // Builds the network from a genome and config. The config supplies the
    // activation function to use for all hidden and output nodes.
    Network(const Genome& genome, const Config& cfg);

    // Runs one forward pass. `inputs` must have exactly num_inputs values
    // (excluding bias — that is handled internally). Returns output node
    // activations in the order they appear in the genome.
    std::vector<double> activate(const std::vector<double>& inputs) const;

private:
    struct Connection {
        uint32_t from;
        uint32_t to;
        double weight;
    };

    uint32_t num_inputs_;
    uint32_t num_outputs_;

    // Node IDs in activation order (topologically sorted).
    std::vector<uint32_t> activation_order_;

    // Incoming connections per node, indexed by node ID.
    // incoming_[node_id] holds all enabled connections whose `to == node_id`.
    // This makes the forward pass O(N + E) instead of O(N * E).
    std::vector<std::vector<Connection>> incoming_;

    // Maps node ID -> index into the value buffer used during activation.
    std::vector<uint32_t> node_index_;

    // Node types mirrored from the genome for activation logic (bias = 1.0,
    // input = raw input, others get the activation function applied).
    std::vector<NodeType> node_types_;

    const Config* cfg_;
};

} // namespace neat
