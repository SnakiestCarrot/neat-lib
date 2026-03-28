#include "neat/genome.hpp"
#include "neat/config.hpp"
#include "neat/innovation.hpp"
#include "neat/random.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <unordered_map>

namespace neat {

// ===========================================================================
// Helper: pack (from, to) into a single uint64_t key
// ===========================================================================

static inline uint64_t pack_pair(uint32_t from, uint32_t to) {
    return (static_cast<uint64_t>(from) << 32) | to;
}

// ===========================================================================
// Construction
// ===========================================================================

Genome Genome::create_minimal(
    uint32_t num_inputs,
    uint32_t num_outputs,
    Random& rng,
    InnovationTracker& innovations
) {
    Genome g;
    g.num_inputs_ = num_inputs;
    g.num_outputs_ = num_outputs;

    uint32_t node_id = 0;

    // Bias node (always id 0)
    g.nodes.push_back({node_id++, NodeType::BIAS});

    // Input nodes
    for (uint32_t i = 0; i < num_inputs; ++i) {
        g.nodes.push_back({node_id++, NodeType::INPUT});
    }

    // Output nodes
    uint32_t first_output = node_id;
    for (uint32_t i = 0; i < num_outputs; ++i) {
        g.nodes.push_back({node_id++, NodeType::OUTPUT});
    }

    // Fully connect all input + bias nodes to all output nodes
    uint32_t num_source = num_inputs + 1; // +1 for bias
    for (uint32_t src = 0; src < num_source; ++src) {
        for (uint32_t dst = 0; dst < num_outputs; ++dst) {
            ConnectionGene conn{
                innovations.get_or_assign(src, first_output + dst),
                src,
                first_output + dst,
                rng.random_double() * 4.0 - 2.0, // [-2.0, 2.0)
                true
            };
            g.add_connection(conn);
        }
    }

    return g;
}

// Binary search lookup for a node by ID, since nodes are kept sorted by id. 
// Returns nullptr if not found.
const NodeGene* Genome::find_node(uint32_t id) const {
    int l = 0;
    int r = static_cast<int>(nodes.size()) - 1;
    while (l <= r) {
        int m = l + (r-l) / 2;
        if (nodes[m].id == id) {
            return &nodes[m];
        } else if (nodes[m].id < id) {
            l = m + 1;
        } else {
            r = m - 1;
        }
    }
    return nullptr;
}

// ===========================================================================
// Mutation
// ===========================================================================

void Genome::mutate(const Config& cfg, Random& rng, InnovationTracker& innovations) {
    if (rng.prob(cfg.prob_mutate_weight)) {
        mutate_weights(cfg, rng);
    }
    if (rng.prob(cfg.prob_add_node)) {
        mutate_add_node(rng, innovations);
    }
    if (rng.prob(cfg.prob_add_link)) {
        mutate_add_connection(cfg, rng, innovations);
    }
    if (rng.prob(cfg.prob_toggle_enable)) {
        mutate_toggle_enable(rng);
    }
}

void Genome::mutate_weights(const Config& cfg, Random& rng) {
    for (auto& conn : connections) {
        if (rng.prob(cfg.prob_weight_replaced)) {
            conn.weight = rng.random_double() * 4.0 - 2.0;
        } else {
            conn.weight += (rng.random_double() * 2.0 - 1.0) * cfg.weight_mutation_power;
        }
    }
}

void Genome::mutate_add_node(Random& rng, InnovationTracker& innovations) {
    // Pick a random enabled connection to split
    std::vector<size_t> enabled_indices;
    for (size_t i = 0; i < connections.size(); ++i) {
        if (connections[i].enabled) {
            enabled_indices.push_back(i);
        }
    }
    if (enabled_indices.empty()) return;

    size_t idx = enabled_indices[rng.random_int(0, static_cast<int>(enabled_indices.size()) - 1)];
    connections[idx].enabled = false;

    // Copy values out before any add_connection call — add_connection inserts
    // into the connections vector, which can reallocate it and invalidate any
    // reference or pointer into it.
    const uint32_t from   = connections[idx].from;
    const uint32_t to     = connections[idx].to;
    const double   weight = connections[idx].weight;

    // New hidden node — use back().id + 1 rather than nodes.size() because
    // crossover can produce genomes with gaps in node IDs (e.g. [0,1,2,5]),
    // and nodes.size() would then collide with an existing ID.
    uint32_t new_node_id = nodes.back().id + 1;
    nodes.push_back({new_node_id, NodeType::HIDDEN});

    // Source -> new node (weight 1.0 to preserve signal)
    add_connection({innovations.get_or_assign(from, new_node_id), from, new_node_id, 1.0, true});

    // New node -> old target (old weight to preserve behavior)
    add_connection({innovations.get_or_assign(new_node_id, to), new_node_id, to, weight, true});
}

void Genome::mutate_add_connection(const Config& cfg, Random& rng, InnovationTracker& innovations) {
    // Build source/target candidate lists
    std::vector<uint32_t> sources;
    std::vector<uint32_t> targets;
    for (const auto& n : nodes) {
        if (n.type != NodeType::OUTPUT) {
            sources.push_back(n.id);
        }
        if (n.type != NodeType::INPUT && n.type != NodeType::BIAS) {
            targets.push_back(n.id);
        }
    }
    if (sources.empty() || targets.empty()) return;

    for (uint32_t attempt = 0; attempt < cfg.max_attempts_add_link; ++attempt) {
        uint32_t from_id = sources[rng.random_int(0, static_cast<int>(sources.size()) - 1)];
        uint32_t to_id = targets[rng.random_int(0, static_cast<int>(targets.size()) - 1)];

        if (from_id == to_id) continue;
        if (has_connection(from_id, to_id)) continue;
        if (would_create_cycle(from_id, to_id)) continue;

        add_connection({innovations.get_or_assign(from_id, to_id), from_id, to_id, rng.random_double() * 4.0 - 2.0, true});
        return;
    }
}

void Genome::mutate_toggle_enable(Random& rng) {
    if (connections.empty()) return;
    int idx = rng.random_int(0, static_cast<int>(connections.size()) - 1);
    connections[idx].enabled = !connections[idx].enabled;
}

Genome Genome::crossover(
    const Genome& more_fit,
    const Genome& less_fit,
    const Config& cfg,
    Random& rng
) {
    Genome child;
    child.num_inputs_ = more_fit.num_inputs_;
    child.num_outputs_ = more_fit.num_outputs_;

    // Index less_fit connections by innovation for O(1) matching
    std::unordered_map<uint32_t, const ConnectionGene*> less_fit_map;
    for (const auto& c : less_fit.connections) {
        less_fit_map[c.innovation] = &c;
    }

    // Track which node IDs the child needs
    std::unordered_set<uint32_t> child_node_ids;

    for (const auto& gene : more_fit.connections) {
        auto it = less_fit_map.find(gene.innovation);

        if (it != less_fit_map.end()) {
            // Matching gene: randomly inherit from either parent
            const ConnectionGene& chosen = rng.prob(0.5) ? gene : *(it->second);
            ConnectionGene child_gene = chosen;

            // If disabled in either parent, chance to re-enable
            if (!gene.enabled || !it->second->enabled) {
                child_gene.enabled = rng.prob(cfg.prob_reenable_gene);
            }

            child.add_connection(child_gene);
        } else {
            // Disjoint/excess: inherit from more fit parent
            child.add_connection(gene);
        }

        child_node_ids.insert(child.connections.back().from);
        child_node_ids.insert(child.connections.back().to);
    }

    // Inherit node genes from the more fit parent for all referenced nodes
    for (const auto& node : more_fit.nodes) {
        if (child_node_ids.count(node.id) || node.type == NodeType::INPUT
            || node.type == NodeType::BIAS || node.type == NodeType::OUTPUT) {
            child.nodes.push_back(node);
        }
    }

    return child;
}

// ===========================================================================
// Compatibility Distance
// ===========================================================================

double Genome::compatibility_distance(
    const Genome& a,
    const Genome& b,
    const Config& cfg
) {
    // Both connection vectors are sorted by innovation number,
    // so we do a single linear merge pass.
    size_t i = 0, j = 0;
    uint32_t excess = 0;
    uint32_t disjoint = 0;
    double weight_diff_sum = 0.0;
    uint32_t matching = 0;

    while (i < a.connections.size() && j < b.connections.size()) {
        uint32_t innov_a = a.connections[i].innovation;
        uint32_t innov_b = b.connections[j].innovation;

        if (innov_a == innov_b) {
            weight_diff_sum += std::abs(a.connections[i].weight - b.connections[j].weight);
            ++matching;
            ++i;
            ++j;
        } else if (innov_a < innov_b) {
            ++disjoint;
            ++i;
        } else {
            ++disjoint;
            ++j;
        }
    }

    // Remaining genes in whichever genome is longer are excess
    excess = static_cast<uint32_t>((a.connections.size() - i) + (b.connections.size() - j));

    // N = gene count of the larger genome (1 if both small, per Stanley's paper)
    double n = static_cast<double>(std::max(a.connections.size(), b.connections.size()));
    if (n < 1.0) n = 1.0;

    double avg_weight_diff = (matching > 0) ? (weight_diff_sum / matching) : 0.0;

    return (cfg.c1 * excess + cfg.c2 * disjoint) / n + cfg.c3 * avg_weight_diff;
}

// ===========================================================================
// Private Helpers
// ===========================================================================

void Genome::add_connection(const ConnectionGene& conn) {
    connection_set_.insert(pack_pair(conn.from, conn.to));

    // Insert while maintaining sorted order by innovation number
    auto it = std::lower_bound(
        connections.begin(), connections.end(), conn.innovation,
        [](const ConnectionGene& c, uint32_t val) { return c.innovation < val; }
    );
    connections.insert(it, conn);
}

bool Genome::has_connection(uint32_t from, uint32_t to) const {
    return connection_set_.count(pack_pair(from, to)) > 0;
}

bool Genome::would_create_cycle(uint32_t from, uint32_t to) const {
    // BFS from 'to' following existing connections.
    // If we can reach 'from', adding from->to would create a cycle.
    std::unordered_set<uint32_t> visited;
    std::queue<uint32_t> frontier;
    frontier.push(to);
    visited.insert(to);

    while (!frontier.empty()) {
        uint32_t current = frontier.front();
        frontier.pop();

        for (const auto& c : connections) {
            // Check all connections, not just enabled ones. A disabled connection
            // can be re-enabled at any time (crossover, toggle mutation), so we
            // must treat the full structural graph as permanent when checking for
            // cycles. Only considering enabled connections here allowed cycles to
            // form when a disabled gene was later re-enabled.
            if (c.from == current) {
                if (c.to == from) return true;
                if (visited.insert(c.to).second) {
                    frontier.push(c.to);
                }
            }
        }
    }

    return false;
}

} // namespace neat
