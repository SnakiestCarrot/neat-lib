#pragma once

#include "neat/population.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace neat {

// Write per-generation telemetry to a CSV file.
// Columns: seed, generation, best_fitness, mean_fitness, worst_fitness,
//          num_species, time_eval_ms, time_speciate_ms, time_reproduce_ms, time_total_ms
void write_csv(const std::string& path,
               const std::vector<GenerationResult>& results,
               uint64_t seed);

// Parse --csv or --csv=<path> from argv.
// Returns the output path, or an empty string if the flag is absent.
// default_name is used when --csv is given without a value.
std::string parse_csv_arg(int argc, char* argv[], const std::string& default_name);

} // namespace neat
