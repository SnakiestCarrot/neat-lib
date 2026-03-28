#include "neat/network.hpp"
#include "neat/config.hpp"

#include <cassert>
#include <cmath>
#include <queue>
#include <stdexcept>
#include <unordered_map>

namespace neat {

// ---------------------------------------------------------------------------
// Activation functions
// ---------------------------------------------------------------------------

static double apply_activation(double x, ActivationType type) {
    switch (type) {
        case ActivationType::SIGMOID:    return 1.0 / (1.0 + std::exp(-x));
        case ActivationType::TANH:       return std::tanh(x);
        case ActivationType::RELU:       return x > 0.0 ? x : 0.0;
        case ActivationType::LEAKY_RELU: return x > 0.0 ? x : 0.01 * x;
    }
    return x; // unreachable
}

// ---------------------------------------------------------------------------
// Construction — topological sort via Kahn's algorithm
// ---------------------------------------------------------------------------

Network::Network(const Genome& genome, const Config& cfg)
    : num_inputs_(genome.num_inputs())
    , num_outputs_(genome.num_outputs())
    , cfg_(&cfg)
{
    const auto& nodes = genome.nodes;
    const auto& conns = genome.connections;

    // Assign a compact index to each node for the value buffer.
    node_index_.resize(nodes.back().id + 1, UINT32_MAX);
    node_types_.resize(nodes.back().id + 1);
    for (uint32_t i = 0; i < static_cast<uint32_t>(nodes.size()); ++i) {
        node_index_[nodes[i].id] = i;
        node_types_[nodes[i].id] = nodes[i].type;
    }

    // Build incoming adjacency list and in-degree counts for Kahn's.
    uint32_t max_id = nodes.back().id;
    incoming_.resize(max_id + 1);

    std::unordered_map<uint32_t, uint32_t> in_degree;
    for (const auto& n : nodes) {
        in_degree[n.id] = 0;
    }
    for (const auto& c : conns) {
        if (!c.enabled) continue;
        incoming_[c.to].push_back({c.from, c.to, c.weight});
        in_degree[c.to]++;
    }

    // Build outgoing adjacency list for Kahn's — maps node_id -> list of targets.
    std::vector<std::vector<uint32_t>> outgoing(max_id + 1);
    for (uint32_t target = 0; target < static_cast<uint32_t>(incoming_.size()); ++target) {
        for (const auto& c : incoming_[target]) {
            outgoing[c.from].push_back(c.to);
        }
    }

    // Kahn's algorithm: seed with all zero-in-degree nodes (INPUT and BIAS).
    std::queue<uint32_t> ready;
    for (const auto& n : nodes) {
        if (in_degree[n.id] == 0) {
            ready.push(n.id);
        }
    }

    while (!ready.empty()) {
        uint32_t node_id = ready.front();
        ready.pop();
        activation_order_.push_back(node_id);

        for (uint32_t target : outgoing[node_id]) {
            if (--in_degree[target] == 0) {
                ready.push(target);
            }
        }
    }

    if (activation_order_.size() != nodes.size()) {
        throw std::runtime_error("Network::Network: cycle detected in genome graph");
    }
}

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------

std::vector<double> Network::activate(const std::vector<double>& inputs) const {
    if (inputs.size() != num_inputs_) {
        throw std::invalid_argument("Network::activate: wrong number of inputs");
    }

    // Value buffer indexed by node_index_.
    std::vector<double> values(node_index_.size(), 0.0);

    // Seed inputs and bias. The genome always places bias at id 0, then inputs.
    for (uint32_t id = 0; id < static_cast<uint32_t>(node_types_.size()); ++id) {
        if (node_index_[id] == UINT32_MAX) continue;
        if (node_types_[id] == NodeType::BIAS) {
            values[id] = 1.0;
        }
    }
    // Input nodes are ids 1..num_inputs_ (bias is 0).
    for (uint32_t i = 0; i < num_inputs_; ++i) {
        values[i + 1] = inputs[i];
    }

    // Propagate in topological order.
    for (uint32_t node_id : activation_order_) {
        NodeType type = node_types_[node_id];
        if (type == NodeType::BIAS || type == NodeType::INPUT) continue;

        double sum = 0.0;
        for (const auto& c : incoming_[node_id]) {
            sum += values[c.from] * c.weight;
        }
        values[node_id] = apply_activation(sum, cfg_->activation);
    }

    // Collect outputs in genome order (OUTPUT nodes after bias+inputs).
    std::vector<double> outputs;
    outputs.reserve(num_outputs_);
    for (const auto& id : activation_order_) {
        if (node_types_[id] == NodeType::OUTPUT) {
            outputs.push_back(values[id]);
        }
    }
    return outputs;
}

} // namespace neat
