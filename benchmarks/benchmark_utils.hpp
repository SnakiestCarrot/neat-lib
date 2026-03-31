#pragma once

#include "../envs/snake/snake_game.hpp"
#include "neat/config.hpp"
#include "neat/network.hpp"
#include <vector>
#include <algorithm> // added for std::max_element

namespace neat {
namespace benchmark_utils {

inline Config get_standard_config() {
    Config cfg;
    cfg.num_inputs = 3;
    cfg.num_outputs = 1;
    cfg.population_size = 150;
    
    cfg.survival_threshold = 0.2;
    cfg.dropoff_age = 15;
    
    cfg.compat_threshold = 3.0;
    cfg.c1 = 1.0;
    cfg.c2 = 1.0;
    cfg.c3 = 0.4;
    
    cfg.prob_mutate_weight = 0.8;
    cfg.prob_weight_replaced = 0.1;
    cfg.prob_add_node = 0.03;
    cfg.prob_add_link = 0.05;
    cfg.prob_toggle_enable = 0.05;
    cfg.prob_crossover = 0.75;
    cfg.prob_reenable_gene = 0.2;
    cfg.weight_mutation_power = 2.5;
    
    cfg.seed = 42; 
    
    return cfg;
}

// New configuration specifically for the snake game
inline Config get_snake_config() {
    Config cfg = get_standard_config();
    cfg.num_inputs = 12;
    cfg.num_outputs = 3;
    return cfg;
}

inline double dummy_eval(Network& net) {
    static const std::vector<double> inputs = {1.0, 0.5, -0.5};
    std::vector<double> outputs = net.activate(inputs);
    return outputs[0];
}

inline double snake_eval(neat::Network& net, uint32_t generation) {
    SnakeGame game(generation);
    while (game.alive()) {
        auto inputs  = game.get_inputs();
        auto outputs = net.activate(inputs);
        int  action  = static_cast<int>(
            std::max_element(outputs.begin(), outputs.end()) - outputs.begin()
        );
        game.step(action);
    }
    double score = static_cast<double>(game.score());
    return game.steps() + score * score * GRID * GRID;
}

} // namespace benchmark_utils
} // namespace neat
