#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

namespace neat {

/**
 * @brief Configuration parameters for the NEAT evolutionary process.
 * Default values are roughly based on Stanley's original 2002 paper.
 */
struct Config {
    // ------------------------------------------------------------------------
    // System Parameters
    // ------------------------------------------------------------------------
    uint64_t seed = 42; // The master seed for the entire evolutionary run
    
    // ------------------------------------------------------------------------
    // Population & Evaluation Parameters
    // ------------------------------------------------------------------------
    unsigned int population_size = 150;
    
    // The proportion of a species allowed to reproduce (e.g., 0.2 means top 20%)
    double survival_threshold = 0.2; 
    
    // Number of generations a species is allowed to stay stagnant before penalization
    unsigned int dropoff_age = 15;

    // ------------------------------------------------------------------------
    // Speciation Parameters (Compatibility Distance formula)
    // distance = (c1 * E / N) + (c2 * D / N) + (c3 * W)
    // ------------------------------------------------------------------------
    double compat_threshold = 3.0; // Distance threshold to separate species
    double c1 = 1.0;               // Importance of Excess genes
    double c2 = 1.0;               // Importance of Disjoint genes
    double c3 = 0.4;               // Importance of average Weight differences

    // ------------------------------------------------------------------------
    // Mutation Probabilities [0.0, 1.0]
    // ------------------------------------------------------------------------
    double prob_mutate_weight    = 0.8;
    double prob_weight_replaced  = 0.1;  // Chance to completely replace a weight instead of perturbing
    double weight_mutation_power = 2.5;  // Maximum perturbation to a weight

    double prob_add_node         = 0.03; // Chance to add a new node
    double prob_add_link         = 0.05; // Chance to add a new connection
    double prob_toggle_enable    = 0.01; // Chance to toggle a connection's enabled status

    // ------------------------------------------------------------------------
    // Crossover Probabilities
    // ------------------------------------------------------------------------
    // Chance that a new offspring is a result of crossover rather than just mutation
    double prob_crossover        = 0.75; 
    // Chance that an inherently disabled gene is re-enabled during crossover
    double prob_reenable_gene    = 0.25; 

    // ------------------------------------------------------------------------
    // Methods
    // ------------------------------------------------------------------------
    
    /**
     * @brief Validates the configuration parameters.
     * @throws std::invalid_argument if any parameters are mathematically invalid.
     */
    void validate() const;
};

} // namespace neat



