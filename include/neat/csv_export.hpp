#pragma once

#include "neat/config.hpp"
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

// Write every relevant field of cfg to a human-readable sidecar file in
// `key = value` format. `env_name` identifies the environment (e.g. "snake").
// Optional extras let callers record env-specific values (max generations,
// solved threshold, num trials, etc.) that are not part of Config.
void write_config(const std::string& path,
                  const std::string& env_name,
                  const Config& cfg,
                  const std::vector<std::pair<std::string, std::string>>& extras = {});

// Parse --csv or --csv=<path> from argv.
// Returns the output path, or an empty string if the flag is absent.
// default_name is used when --csv is given without a value.
std::string parse_csv_arg(int argc, char* argv[], const std::string& default_name);

// Derive a sidecar config path from a csv path: "foo.csv" -> "foo.config.txt".
// If csv_path has no .csv suffix, ".config.txt" is appended.
std::string config_sidecar_path(const std::string& csv_path);

} // namespace neat
