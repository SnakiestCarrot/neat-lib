#include "neat/neat.hpp"

#include <algorithm>
#include <chrono>
#include <deque>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

// ============================================================================
// Constants
// ============================================================================

static constexpr int GRID             = 20;
static constexpr int STAGNATION_LIMIT = GRID * GRID; // steps without food before death

// ============================================================================
// Direction helpers
// ============================================================================

enum Dir { UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3 };

// x and y deltas for each direction (y increases downward)
static constexpr int DX[4] = { 0,  1,  0, -1 };
static constexpr int DY[4] = {-1,  0,  1,  0 };

struct Pos {
    int x, y;
    bool operator==(const Pos& o) const { return x == o.x && y == o.y; }
};

// ============================================================================
// SnakeGame
// ============================================================================

class SnakeGame {
public:
    explicit SnakeGame(uint64_t seed) : rng_(seed) {
        // Start in the middle of the grid, heading right, length 3
        Pos head = { GRID / 2, GRID / 2 };
        body_.push_back(head);
        body_.push_back({ head.x - 1, head.y });
        body_.push_back({ head.x - 2, head.y });
        direction_ = RIGHT;

        for (const auto& p : body_) {
            occupied_[p.y][p.x] = true;
        }

        place_food();
    }

    bool alive() const { return alive_; }
    int  score() const { return score_; }
    int  steps() const { return steps_; }

    const std::deque<Pos>& body() const { return body_; }
    Pos  food()              const { return food_; }
    Dir  direction()         const { return direction_; }

    // Returns 12 inputs — for each of 4 relative directions (forward, right,
    // backward, left): wall distance normalised, food on ray, body on ray.
    std::vector<double> get_inputs() const {
        std::vector<double> inputs;
        inputs.reserve(12);

        Pos head = body_.front();

        for (int rel = 0; rel < 4; ++rel) {
            int dir = (direction_ + rel) % 4;

            bool food_seen = false;
            bool body_seen = false;
            int  dist      = 0;
            int  cx        = head.x + DX[dir];
            int  cy        = head.y + DY[dir];

            while (cx >= 0 && cx < GRID && cy >= 0 && cy < GRID) {
                ++dist;
                if (!food_seen && cx == food_.x && cy == food_.y) {
                    food_seen = true;
                }
                if (!body_seen && occupied_[cy][cx]) {
                    body_seen = true;
                }
                cx += DX[dir];
                cy += DY[dir];
            }
            // dist is now the number of steps to the wall

            inputs.push_back(static_cast<double>(dist) / GRID);
            inputs.push_back(food_seen ? 1.0 : 0.0);
            inputs.push_back(body_seen ? 1.0 : 0.0);
        }

        return inputs;
    }

    // Action: 0 = turn left, 1 = go straight, 2 = turn right (relative)
    void step(int action) {
        if (!alive_) return;

        if      (action == 0) direction_ = static_cast<Dir>((direction_ + 3) % 4);
        else if (action == 2) direction_ = static_cast<Dir>((direction_ + 1) % 4);

        Pos head = body_.front();
        Pos next = { head.x + DX[direction_], head.y + DY[direction_] };

        // Wall collision
        if (next.x < 0 || next.x >= GRID || next.y < 0 || next.y >= GRID) {
            alive_ = false;
            return;
        }

        // Self collision — the tail will vacate this step so it's safe to
        // move into it, unless we're about to eat food (tail won't move then)
        bool eats_food = (next == food_);
        if (occupied_[next.y][next.x]) {
            bool is_tail = (next == body_.back());
            if (!is_tail || eats_food) {
                alive_ = false;
                return;
            }
        }

        // Advance head
        body_.push_front(next);
        occupied_[next.y][next.x] = true;

        if (eats_food) {
            ++score_;
            steps_since_food_ = 0;
            if (static_cast<int>(body_.size()) < GRID * GRID) {
                place_food();
            }
        } else {
            // Remove tail
            occupied_[body_.back().y][body_.back().x] = false;
            body_.pop_back();
        }

        ++steps_;
        if (++steps_since_food_ >= STAGNATION_LIMIT) {
            alive_ = false;
        }
    }

private:
    void place_food() {
        int empty = GRID * GRID - static_cast<int>(body_.size());
        if (empty == 0) return;

        int target = rng_.random_int(0, empty - 1);
        int idx    = 0;
        for (int y = 0; y < GRID; ++y) {
            for (int x = 0; x < GRID; ++x) {
                if (!occupied_[y][x]) {
                    if (idx++ == target) {
                        food_ = { x, y };
                        return;
                    }
                }
            }
        }
    }

    neat::Random    rng_;
    std::deque<Pos> body_;
    bool            occupied_[GRID][GRID] = {};
    Pos             food_             = {};
    Dir             direction_        = RIGHT;
    int             score_            = 0;
    int             steps_            = 0;
    int             steps_since_food_ = 0;
    bool            alive_            = true;
};

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

int main() {
    neat::Config cfg;
    cfg.num_inputs      = 12;  // 3 signals x 4 relative directions
    cfg.num_outputs     = 3;   // turn left, straight, turn right
    cfg.population_size = 30;
    cfg.seed            = 42;

    std::cout << "=== NEAT Snake " << GRID << "x" << GRID << " ===\n"
              << "Population: " << cfg.population_size
              << "  |  Inputs: "  << cfg.num_inputs
              << "  |  Outputs: " << cfg.num_outputs << "\n\n";

    neat::Population pop(cfg);

    pop.run_until(
        [&](neat::Network& net) {
            return evaluate(net, pop.generation());
        },
        [&](const neat::GenerationResult& r) {
            print_stats(r);

            if (r.generation % 1000 == 0) {
                auto best = pop.best_network();
                visualize(best, r.generation);
            }

            return r.generation >= 10000;
        }
    );

    std::cout << "\n=== Training complete ===\n"
              << "Showing final best network...\n\n";
    auto best = pop.best_network();
    visualize(best, pop.generation() - 1);

    return 0;
}
