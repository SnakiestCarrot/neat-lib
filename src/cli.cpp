#include "neat/cli.hpp"

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace neat {

void parse_config_args(Config& cfg, int argc, char* argv[]) {
    auto next_val = [&](int i) -> const char* {
        if (i + 1 >= argc) {
            throw std::invalid_argument(std::string("missing value for ") + argv[i]);
        }
        return argv[i + 1];
    };

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];

        if (std::strcmp(arg, "--seed") == 0) {
            cfg.seed = static_cast<uint64_t>(std::stoull(next_val(i)));
            ++i;
        } else if (std::strcmp(arg, "--population-size") == 0) {
            cfg.population_size = static_cast<unsigned int>(std::stoul(next_val(i)));
            ++i;
        } else if (std::strcmp(arg, "--compat-threshold") == 0) {
            cfg.compat_threshold = std::stod(next_val(i));
            ++i;
        } else if (std::strcmp(arg, "--dropoff-age") == 0) {
            cfg.dropoff_age = static_cast<unsigned int>(std::stoul(next_val(i)));
            ++i;
        } else if (std::strcmp(arg, "--survival-threshold") == 0) {
            cfg.survival_threshold = std::stod(next_val(i));
            ++i;
        } else if (std::strcmp(arg, "--prob-add-node") == 0) {
            cfg.prob_add_node = std::stod(next_val(i));
            ++i;
        } else if (std::strcmp(arg, "--prob-add-link") == 0) {
            cfg.prob_add_link = std::stod(next_val(i));
            ++i;
        } else if (std::strcmp(arg, "--prob-mutate-weight") == 0) {
            cfg.prob_mutate_weight = std::stod(next_val(i));
            ++i;
        } else if (std::strcmp(arg, "--parallel-eval") == 0) {
            cfg.parallel_eval = true;
        } else if (std::strcmp(arg, "--no-parallel-eval") == 0) {
            cfg.parallel_eval = false;
        } else if (std::strcmp(arg, "--activation") == 0) {
            const char* val = next_val(i);
            ++i;
            if (std::strcmp(val, "sigmoid") == 0) {
                cfg.activation = ActivationType::SIGMOID;
            } else if (std::strcmp(val, "tanh") == 0) {
                cfg.activation = ActivationType::TANH;
            } else if (std::strcmp(val, "relu") == 0) {
                cfg.activation = ActivationType::RELU;
            } else if (std::strcmp(val, "leaky_relu") == 0) {
                cfg.activation = ActivationType::LEAKY_RELU;
            } else {
                throw std::invalid_argument(std::string("unknown activation: ") + val);
            }
        }
        // Unknown flags (e.g. --csv) are silently skipped.
    }
}

} // namespace neat
