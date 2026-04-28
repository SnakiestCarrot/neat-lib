#include "neat/csv_export.hpp"

#include <chrono>
#include <ctime>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace neat {

namespace {

const char* activation_name(ActivationType a) {
    switch (a) {
        case ActivationType::SIGMOID:    return "sigmoid";
        case ActivationType::TANH:       return "tanh";
        case ActivationType::RELU:       return "relu";
        case ActivationType::LEAKY_RELU: return "leaky_relu";
    }
    return "unknown";
}

std::string iso_timestamp_now() {
    auto now = std::chrono::system_clock::now();
    auto t   = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    std::ostringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return ss.str();
}

} // namespace

void write_csv(const std::string& path,
               const std::vector<GenerationResult>& results,
               uint64_t seed)
{
    std::ofstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("write_csv: cannot open " + path);
    }

    f << "seed,generation,best_fitness,mean_fitness,worst_fitness,"
         "num_species,time_eval_ms,time_speciate_ms,time_reproduce_ms,time_total_ms\n";

    for (const auto& r : results) {
        f << seed           << ','
          << r.generation   << ','
          << r.best_fitness << ','
          << r.mean_fitness << ','
          << r.worst_fitness << ','
          << r.num_species  << ','
          << r.time_eval_ms << ','
          << r.time_speciate_ms << ','
          << r.time_reproduce_ms << ','
          << r.time_total_ms << '\n';
    }
}

void write_config(const std::string& path,
                  const std::string& env_name,
                  const Config& cfg,
                  const std::vector<std::pair<std::string, std::string>>& extras)
{
    std::ofstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("write_config: cannot open " + path);
    }

    // 10 sig figs is enough to round-trip the typical NEAT params cleanly
    // while keeping common values like 0.03 / 0.2 readable.
    f << std::setprecision(10);

    f << "# NEAT run configuration\n";
    f << "# Written: " << iso_timestamp_now() << "\n";
    f << "\n";

    f << "[run]\n";
    f << "env                    = " << env_name << "\n";
    f << "seed                   = " << cfg.seed << "\n";
    f << "\n";

    f << "[topology]\n";
    f << "num_inputs             = " << cfg.num_inputs  << "\n";
    f << "num_outputs            = " << cfg.num_outputs << "\n";
    f << "activation             = " << activation_name(cfg.activation) << "\n";
    f << "\n";

    f << "[population]\n";
    f << "population_size        = " << cfg.population_size    << "\n";
    f << "survival_threshold     = " << cfg.survival_threshold << "\n";
    f << "dropoff_age            = " << cfg.dropoff_age        << "\n";
    f << "\n";

    f << "[speciation]\n";
    f << "compat_threshold       = " << cfg.compat_threshold << "\n";
    f << "c1                     = " << cfg.c1 << "\n";
    f << "c2                     = " << cfg.c2 << "\n";
    f << "c3                     = " << cfg.c3 << "\n";
    f << "\n";

    f << "[mutation]\n";
    f << "prob_mutate_weight     = " << cfg.prob_mutate_weight    << "\n";
    f << "prob_weight_replaced   = " << cfg.prob_weight_replaced  << "\n";
    f << "weight_mutation_power  = " << cfg.weight_mutation_power << "\n";
    f << "prob_add_node          = " << cfg.prob_add_node         << "\n";
    f << "prob_add_link          = " << cfg.prob_add_link         << "\n";
    f << "max_attempts_add_link  = " << cfg.max_attempts_add_link << "\n";
    f << "prob_toggle_enable     = " << cfg.prob_toggle_enable    << "\n";
    f << "\n";

    f << "[crossover]\n";
    f << "prob_crossover         = " << cfg.prob_crossover     << "\n";
    f << "prob_reenable_gene     = " << cfg.prob_reenable_gene << "\n";
    f << "\n";

    f << "[parallelism]\n";
    f << "parallel_eval          = " << (cfg.parallel_eval ? "true" : "false") << "\n";

    if (!extras.empty()) {
        f << "\n[extras]\n";
        for (const auto& [k, v] : extras) {
            f << k << " = " << v << "\n";
        }
    }
}

std::string config_sidecar_path(const std::string& csv_path)
{
    const std::string suffix = ".csv";
    if (csv_path.size() >= suffix.size() &&
        csv_path.compare(csv_path.size() - suffix.size(), suffix.size(), suffix) == 0)
    {
        return csv_path.substr(0, csv_path.size() - suffix.size()) + ".config.txt";
    }
    return csv_path + ".config.txt";
}

std::string parse_csv_arg(int argc, char* argv[], const std::string& default_name)
{
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (std::strncmp(arg, "--csv=", 6) == 0) {
            return std::string(arg + 6);
        }
        if (std::strcmp(arg, "--csv") == 0) {
            return default_name;
        }
    }
    return {};
}

} // namespace neat
