#include "neat/neat.hpp"
#include "neat/csv_export.hpp"
#include "snake_game.hpp"

#include <algorithm>
#include <chrono>
#include <deque>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

// ============================================================================
// Evaluation
//
// All genomes in the same generation share the same seed so they face
// identical food sequences — fitness comparison is fair across the population.
// ============================================================================

static double evaluate(neat::Network& net, uint32_t generation) {
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

// ============================================================================
// Visualization
// ============================================================================

static void print_stats(const neat::GenerationResult& r) {
    std::cout
        << "Gen "     << std::setw(4) << r.generation
        << " | Best: " << std::setw(10) << std::fixed << std::setprecision(1) << r.best_fitness
        << " | Mean: " << std::setw(10) << r.mean_fitness
        << " | Species: " << r.num_species
        << "\n";
}

static void visualize(neat::Network& net, uint32_t generation) {
    SnakeGame game(generation);

    while (game.alive()) {
        // Clear terminal and move cursor to top-left
        std::cout << "\033[2J\033[H";

        // Build grid
        char grid[GRID][GRID];
        for (int y = 0; y < GRID; ++y)
            for (int x = 0; x < GRID; ++x)
                grid[y][x] = ' ';

        grid[game.food().y][game.food().x] = '*';

        const auto& body = game.body();
        for (size_t i = body.size() - 1; i > 0; --i) {
            grid[body[i].y][body[i].x] = 'o';
        }
        grid[body.front().y][body.front().x] = 'O';

        // Print
        std::cout << "Generation " << generation
                  << "  |  Score: " << game.score()
                  << "  |  Steps: " << game.steps() << "\n";
        std::cout << "+" << std::string(GRID, '-') << "+\n";
        for (int y = 0; y < GRID; ++y) {
            std::cout << "|";
            for (int x = 0; x < GRID; ++x) std::cout << grid[y][x];
            std::cout << "|\n";
        }
        std::cout << "+" << std::string(GRID, '-') << "+\n";
        std::cout << "O = head  o = body  * = food\n";
        std::cout << std::flush;

        auto inputs  = game.get_inputs();
        auto outputs = net.activate(inputs);
        int  action  = static_cast<int>(
            std::max_element(outputs.begin(), outputs.end()) - outputs.begin()
        );
        game.step(action);

        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }

    std::cout << "\033[2J\033[H";
    std::cout << "Game over — Score: " << game.score()
              << "  |  Steps: " << game.steps() << "\n\n";
    std::this_thread::sleep_for(std::chrono::seconds(2));
}

// ============================================================================
// Main
// ============================================================================

static bool has_flag(int argc, char* argv[], const char* flag) {
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == flag) return true;
    return false;
}

int main(int argc, char* argv[]) {
    std::string csv_path = neat::parse_csv_arg(argc, argv, "snake_results.csv");
    bool visualize_on    = !has_flag(argc, argv, "--no-viz");
    neat::Config cfg;
    cfg.num_inputs      = 12;  // 3 signals x 4 relative directions
    cfg.num_outputs     = 3;   // turn left, straight, turn right
    cfg.population_size = 30;
    cfg.seed            = 42;

    neat::parse_config_args(cfg, argc, argv);

    constexpr int MAX_GENS = 1000;

    if (!csv_path.empty()) {
        neat::write_config(neat::config_sidecar_path(csv_path), "snake", cfg, {
            {"max_generations", std::to_string(MAX_GENS)},
            {"grid_size",       std::to_string(GRID)},
        });
    }

    std::cout << "=== NEAT Snake " << GRID << "x" << GRID << " ===\n"
              << "Population: " << cfg.population_size
              << "  |  Inputs: "  << cfg.num_inputs
              << "  |  Outputs: " << cfg.num_outputs << "\n\n";

    neat::Population pop(cfg);

    auto run = pop.run_until(
        [&](neat::Network& net) {
            return evaluate(net, pop.generation());
        },
        [&](const neat::GenerationResult& r) {
            print_stats(r);

            if (visualize_on && r.generation % 1000 == 0) {
                auto best = pop.best_network();
                visualize(best, r.generation);
            }

            return r.generation >= MAX_GENS;
        }
    );

    if (!csv_path.empty()) {
        neat::write_csv(csv_path, run.generations, cfg.seed);
        std::cout << "Wrote " << csv_path << "\n";
    }

    std::cout << "\n=== Training complete ===\n";
    if (visualize_on) {
        std::cout << "Showing final best network...\n\n";
        auto best = pop.best_network();
        visualize(best, pop.generation() - 1);
    }

    return 0;
}
