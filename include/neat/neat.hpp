#pragma once

// NEAT — NeuroEvolution of Augmenting Topologies
//
// Include this header to access the full library.
//
// Typical usage:
//
//   #include "neat/neat.hpp"
//
//   neat::Config cfg;
//   cfg.num_inputs  = 4;
//   cfg.num_outputs = 2;
//
//   neat::Population pop(cfg);
//
//   auto result = pop.run_until(
//       [](neat::Network& net) {
//           auto output = net.activate({1.0, 0.5, -1.0, 0.2});
//           return compute_fitness(output);
//       },
//       [](const neat::GenerationResult& r) {
//           return r.best_fitness >= 1000.0 || r.generation >= 500;
//       }
//   );

#include "neat/cli.hpp"
#include "neat/config.hpp"
#include "neat/csv_export.hpp"
#include "neat/genome.hpp"
#include "neat/innovation.hpp"
#include "neat/network.hpp"
#include "neat/population.hpp"
#include "neat/random.hpp"
