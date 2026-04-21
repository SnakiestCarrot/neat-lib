#include "neat/csv_export.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

namespace neat {

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
