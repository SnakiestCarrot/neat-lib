#include "../include/neat/config.hpp"

namespace neat {

void Config::validate() const {
    // Population checks
    if (population_size == 0) {
        throw std::invalid_argument("NEAT Config: population_size must be greater than 0.");
    }
    
    if (survival_threshold <= 0.0 || survival_threshold > 1.0) {
        throw std::invalid_argument("NEAT Config: survival_threshold must be in range (0.0, 1.0].");
    }

    // Speciation checks
    if (compat_threshold <= 0.0) {
        throw std::invalid_argument("NEAT Config: compat_threshold must be strictly positive.");
    }
    if (c1 < 0.0 || c2 < 0.0 || c3 < 0.0) {
        throw std::invalid_argument("NEAT Config: Speciation coefficients (c1, c2, c3) cannot be negative.");
    }

    // Probability checks helper lambda
    auto check_prob = [](double p, const std::string& name) {
        if (p < 0.0 || p > 1.0) {
            throw std::invalid_argument("NEAT Config: " + name + " must be between 0.0 and 1.0.");
        }
    };

    check_prob(prob_mutate_weight, "prob_mutate_weight");
    check_prob(prob_weight_replaced, "prob_weight_replaced");
    check_prob(prob_add_node, "prob_add_node");
    check_prob(prob_add_link, "prob_add_link");
    check_prob(prob_toggle_enable, "prob_toggle_enable");
    check_prob(prob_crossover, "prob_crossover");
    check_prob(prob_reenable_gene, "prob_reenable_gene");
    
    // Perturbation check
    if (weight_mutation_power <= 0.0) {
        throw std::invalid_argument("NEAT Config: weight_mutation_power must be positive.");
    }
}

} // namespace neat