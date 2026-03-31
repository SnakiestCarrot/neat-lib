#pragma once

#include "neat/random.hpp"
#include <deque>
#include <vector>
#include <algorithm>

// ============================================================================
// Constants
// ============================================================================
static constexpr int GRID             = 20;
static constexpr int STAGNATION_LIMIT = GRID * GRID;

// ============================================================================
// Direction helpers
// ============================================================================
enum Dir { UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3 };

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

            inputs.push_back(static_cast<double>(dist) / GRID);
            inputs.push_back(food_seen ? 1.0 : 0.0);
            inputs.push_back(body_seen ? 1.0 : 0.0);
        }

        return inputs;
    }

    void step(int action) {
        if (!alive_) return;

        if      (action == 0) direction_ = static_cast<Dir>((direction_ + 3) % 4);
        else if (action == 2) direction_ = static_cast<Dir>((direction_ + 1) % 4);

        Pos head = body_.front();
        Pos next = { head.x + DX[direction_], head.y + DY[direction_] };

        if (next.x < 0 || next.x >= GRID || next.y < 0 || next.y >= GRID) {
            alive_ = false;
            return;
        }

        bool eats_food = (next == food_);
        if (occupied_[next.y][next.x]) {
            bool is_tail = (next == body_.back());
            if (!is_tail || eats_food) {
                alive_ = false;
                return;
            }
        }

        body_.push_front(next);
        occupied_[next.y][next.x] = true;

        if (eats_food) {
            ++score_;
            steps_since_food_ = 0;
            if (static_cast<int>(body_.size()) < GRID * GRID) {
                place_food();
            }
        } else {
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
