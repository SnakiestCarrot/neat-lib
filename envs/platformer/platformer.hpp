#pragma once

#include "neat/random.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace env {

// ============================================================================
// Tile types — all behave identically for collision.
// Distinction is purely visual (for the HTML viz).
// ============================================================================

enum class Tile : uint8_t { AIR = 0, GROUND, BRICK, QUESTION, PIPE, FLAG };

// ============================================================================
// Enemy (Goomba)
//
// Walks back and forth on the ground.  Reverses at walls and edges.
// Mario dies on side contact, but can stomp from above (enemy dies,
// Mario gets a small upward bounce).
// ============================================================================

struct Enemy {
    double x = 0.0, y = 0.0;
    double vx = 0.0;
    bool   alive = true;

    static constexpr double W     = 0.80;
    static constexpr double H     = 0.80;
    static constexpr double SPEED = 1.5;   // tiles/s
};

// ============================================================================
// Platformer — 2D side-scroller inspired by Super Mario Bros 1-1
//
// Observations (20):
//   6 ground-ahead   (solid at row 12, for gap detection)
//   6 obstacle-ahead (solid at Mario's body height)
//   2 ceiling-ahead  (solid at Mario's head height)
//   1 y_velocity     (normalised)
//   1 on_ground      (0 or 1)
//   2 nearest enemy  (proximity 0–1, relative height -1–1)
//   2 2nd nearest    (same, 0 if none)
//
// Actions (3 SIGMOID outputs, thresholded at 0.5):
//   [0] left   [1] right   [2] jump
// ============================================================================

struct PlatformerParams {
    // Physics
    double gravity    = 45.0;   // tiles/s²
    double jump_vel   = 16.0;   // tiles/s upward  → jump height ≈ 2.8 tiles, range ≈ 3.9
    double max_speed  = 5.5;    // tiles/s horizontal
    double run_accel  = 18.0;   // tiles/s²
    double friction   = 12.0;   // tiles/s² deceleration when not pressing
    double dt         = 1.0 / 60.0;

    // Mario hitbox (slightly smaller than 1 tile for collision forgiveness)
    double mario_w = 0.80;
    double mario_h = 0.90;

    // Stomp bounce (fraction of jump_vel)
    double stomp_bounce = 0.5;

    // Limits
    int max_steps  = 7000;
    int idle_limit = 180;    // 3 seconds without rightward progress → terminate

    // Level dimensions
    static constexpr int LEVEL_W = 212;
    static constexpr int LEVEL_H = 14;   // rows 0 (top) to 13 (bottom)
};

class Platformer {
public:
    static constexpr int LEVEL_W = PlatformerParams::LEVEL_W;
    static constexpr int LEVEL_H = PlatformerParams::LEVEL_H;
    static constexpr int NUM_OBS = 20;
    static constexpr double ENEMY_SCAN_RANGE = 8.0;  // tiles ahead

    explicit Platformer(const PlatformerParams& p = {}) : p_(p) {
        tiles_.assign(LEVEL_H, std::vector<Tile>(LEVEL_W, Tile::AIR));
        build_level();
    }

    void reset(neat::Random& /*rng*/) {
        x_  = 3.0;
        y_  = ground_row_ - p_.mario_h;   // feet on ground
        vx_ = 0.0;
        vy_ = 0.0;
        max_x_        = x_;
        on_ground_    = true;
        reached_flag_ = false;
        dead_         = false;
        step_         = 0;
        idle_         = 0;
        enemies_      = initial_enemies_;  // respawn all enemies
    }

    // Raw SIGMOID outputs in [0,1].  Each > 0.5 counts as "pressed".
    void step(double left_val, double right_val, double jump_val) {
        bool left  = left_val  > 0.5;
        bool right = right_val > 0.5;
        bool jump  = jump_val  > 0.5;

        // --- horizontal acceleration / friction ---
        if (right && !left) {
            vx_ += p_.run_accel * p_.dt;
        } else if (left && !right) {
            vx_ -= p_.run_accel * p_.dt;
        } else {
            if (vx_ > 0.0) vx_ = std::max(0.0, vx_ - p_.friction * p_.dt);
            else            vx_ = std::min(0.0, vx_ + p_.friction * p_.dt);
        }
        vx_ = std::clamp(vx_, -p_.max_speed, p_.max_speed);

        // --- jump ---
        if (jump && on_ground_) {
            vy_ = -p_.jump_vel;
            on_ground_ = false;
        }

        // --- gravity ---
        vy_ += p_.gravity * p_.dt;

        // --- move & collide on each axis independently ---
        move_axis_x();
        move_axis_y();

        // --- move enemies ---
        move_enemies();

        // --- mario ↔ enemy collisions ---
        check_enemy_collisions();

        // --- bookkeeping ---
        double prev_max = max_x_;
        max_x_ = std::max(max_x_, x_);
        idle_  = (max_x_ > prev_max + 0.01) ? 0 : idle_ + 1;

        if (y_ > LEVEL_H + 2) dead_ = true;
        if (x_ + p_.mario_w * 0.5 >= flag_x_) reached_flag_ = true;
        if (x_ < 0.0) { x_ = 0.0; vx_ = std::max(vx_, 0.0); }

        ++step_;
    }

    std::vector<double> observe() const {
        std::vector<double> obs;
        obs.reserve(NUM_OBS);

        int mx = static_cast<int>(x_ + p_.mario_w * 0.5);

        // 6 columns ahead: solid at ground row?  → gap detection
        for (int dx = 1; dx <= 6; ++dx)
            obs.push_back(is_solid_safe(mx + dx, ground_row_) ? 1.0 : 0.0);

        // 6 columns ahead: solid at body height?  → obstacle detection
        int body_row = static_cast<int>(y_ + p_.mario_h * 0.5);
        for (int dx = 1; dx <= 6; ++dx)
            obs.push_back(is_solid_safe(mx + dx, body_row) ? 1.0 : 0.0);

        // 2 columns ahead: solid at head height?  → ceiling detection
        int head_row = static_cast<int>(y_);
        for (int dx = 1; dx <= 2; ++dx)
            obs.push_back(is_solid_safe(mx + dx, head_row) ? 1.0 : 0.0);

        // vertical velocity (normalised)
        obs.push_back(std::clamp(vy_ / p_.jump_vel, -1.0, 1.0));

        // on ground
        obs.push_back(on_ground_ ? 1.0 : 0.0);

        // --- enemy detection: 2 nearest enemies ahead within scan range ---
        // Collect distances to alive enemies ahead of Mario
        struct EnemyInfo { double dx; double dy; };
        std::vector<EnemyInfo> ahead;
        double mario_cx = x_ + p_.mario_w * 0.5;
        double mario_cy = y_ + p_.mario_h * 0.5;

        for (const auto& e : enemies_) {
            if (!e.alive) continue;
            double ecx = e.x + Enemy::W * 0.5;
            double dx  = ecx - mario_cx;
            if (dx > 0.0 && dx < ENEMY_SCAN_RANGE) {
                double dy = (e.y + Enemy::H * 0.5) - mario_cy;
                ahead.push_back({dx, dy});
            }
        }
        // Sort by distance
        std::sort(ahead.begin(), ahead.end(),
                  [](const EnemyInfo& a, const EnemyInfo& b) {
                      return a.dx < b.dx;
                  });

        // Nearest enemy: proximity (1=touching, 0=far/none), relative height
        for (int i = 0; i < 2; ++i) {
            if (i < static_cast<int>(ahead.size())) {
                obs.push_back(1.0 - ahead[i].dx / ENEMY_SCAN_RANGE);
                obs.push_back(std::clamp(ahead[i].dy / 3.0, -1.0, 1.0));
            } else {
                obs.push_back(0.0);
                obs.push_back(0.0);
            }
        }

        return obs;
    }

    bool terminated() const {
        return dead_
            || reached_flag_
            || step_ >= p_.max_steps
            || idle_ >= p_.idle_limit;
    }

    // --- accessors ---
    double x()             const { return x_; }
    double y()             const { return y_; }
    double vx()            const { return vx_; }
    double vy()            const { return vy_; }
    double max_x()         const { return max_x_; }
    bool   reached_flag()  const { return reached_flag_; }
    bool   is_dead()       const { return dead_; }
    int    step_count()    const { return step_; }
    int    flag_col()      const { return flag_x_; }

    const std::vector<Enemy>& enemies() const { return enemies_; }
    int num_enemies() const { return static_cast<int>(initial_enemies_.size()); }

    // level access (for viz serialisation)
    Tile tile_at(int col, int row) const {
        if (col < 0 || col >= LEVEL_W || row < 0 || row >= LEVEL_H)
            return Tile::AIR;
        return tiles_[row][col];
    }
    int level_width()  const { return LEVEL_W; }
    int level_height() const { return LEVEL_H; }

private:
    // ------------------------------------------------------------------ tiles
    bool is_solid(int col, int row) const {
        Tile t = tiles_[row][col];
        return t != Tile::AIR && t != Tile::FLAG;
    }

    bool is_solid_safe(int col, int row) const {
        if (col < 0 || col >= LEVEL_W || row < 0 || row >= LEVEL_H)
            return false;
        return is_solid(col, row);
    }

    // ------------------------------------------------------ collision helpers

    void move_axis_x() {
        x_ += vx_ * p_.dt;

        int c0 = std::max(0, static_cast<int>(std::floor(x_)));
        int c1 = std::min(LEVEL_W - 1, static_cast<int>(std::floor(x_ + p_.mario_w - 1e-6)));
        int r0 = std::max(0, static_cast<int>(std::floor(y_)));
        int r1 = std::min(LEVEL_H - 1, static_cast<int>(std::floor(y_ + p_.mario_h - 1e-6)));

        for (int r = r0; r <= r1; ++r) {
            for (int c = c0; c <= c1; ++c) {
                if (!is_solid(c, r)) continue;
                if (vx_ > 0.0) {
                    x_  = static_cast<double>(c) - p_.mario_w;
                    vx_ = 0.0;
                } else if (vx_ < 0.0) {
                    x_  = static_cast<double>(c + 1);
                    vx_ = 0.0;
                }
            }
        }
    }

    void move_axis_y() {
        y_ += vy_ * p_.dt;

        int c0 = std::max(0, static_cast<int>(std::floor(x_)));
        int c1 = std::min(LEVEL_W - 1, static_cast<int>(std::floor(x_ + p_.mario_w - 1e-6)));
        int r0 = std::max(0, static_cast<int>(std::floor(y_)));
        int r1 = std::min(LEVEL_H - 1, static_cast<int>(std::floor(y_ + p_.mario_h - 1e-6)));

        on_ground_ = false;
        for (int r = r0; r <= r1; ++r) {
            for (int c = c0; c <= c1; ++c) {
                if (!is_solid(c, r)) continue;
                if (vy_ > 0.0) {
                    y_  = static_cast<double>(r) - p_.mario_h;
                    vy_ = 0.0;
                    on_ground_ = true;
                } else if (vy_ < 0.0) {
                    y_  = static_cast<double>(r + 1);
                    vy_ = 0.0;
                }
            }
        }
    }

    // --------------------------------------------------------- enemy movement

    void move_enemies() {
        for (auto& e : enemies_) {
            if (!e.alive) continue;

            e.x += e.vx * p_.dt;

            // Wall collision: solid tile at enemy body level in movement direction
            int ahead_col = (e.vx > 0)
                ? static_cast<int>(std::floor(e.x + Enemy::W))
                : static_cast<int>(std::floor(e.x)) - 1;
            int body_row = static_cast<int>(std::floor(e.y + Enemy::H * 0.5));
            if (ahead_col < 0 || ahead_col >= LEVEL_W
                || is_solid_safe(ahead_col, body_row)) {
                e.vx = -e.vx;
            }

            // Edge detection: no ground ahead → reverse
            int feet_col = (e.vx > 0)
                ? static_cast<int>(std::floor(e.x + Enemy::W))
                : static_cast<int>(std::floor(e.x));
            if (feet_col < 0 || feet_col >= LEVEL_W
                || !is_solid_safe(feet_col, ground_row_)) {
                e.vx = -e.vx;
            }
        }
    }

    // ------------------------------------------------ mario ↔ enemy collision

    void check_enemy_collisions() {
        for (auto& e : enemies_) {
            if (!e.alive) continue;

            // AABB overlap?
            bool overlap =
                x_ + p_.mario_w > e.x       && x_ < e.x + Enemy::W &&
                y_ + p_.mario_h > e.y       && y_ < e.y + Enemy::H;
            if (!overlap) continue;

            // Stomp: Mario falling and feet above enemy's mid-height
            if (vy_ >= 0.0 && y_ + p_.mario_h <= e.y + Enemy::H * 0.5) {
                e.alive = false;
                vy_ = -p_.jump_vel * p_.stomp_bounce;   // bounce up
            } else {
                dead_ = true;
            }
        }
    }

    // -------------------------------------------------------- level building

    void place_enemy(double ex) {
        Enemy e;
        e.x     = ex;
        e.y     = static_cast<double>(ground_row_) - Enemy::H;
        e.vx    = -Enemy::SPEED;
        e.alive = true;
        initial_enemies_.push_back(e);
    }

    void build_level() {
        // Helpers
        auto ground = [&](int x0, int x1) {
            for (int c = x0; c <= x1; ++c) {
                tiles_[12][c] = Tile::GROUND;
                tiles_[13][c] = Tile::GROUND;
            }
        };
        auto pipe = [&](int x, int height) {
            for (int h = 0; h < height; ++h) {
                int r = 11 - h;
                tiles_[r][x]     = Tile::PIPE;
                tiles_[r][x + 1] = Tile::PIPE;
            }
        };
        auto brick    = [&](int x, int r) { tiles_[r][x] = Tile::BRICK; };
        auto question = [&](int x, int r) { tiles_[r][x] = Tile::QUESTION; };

        // Level design: 15 gaps of 3 tiles each, spaced with varying ground
        // lengths.  Goombas patrol the longer ground sections.

        // ======== Flat start (x=0–24) ========
        ground(0, 24);
        question(12, 8);
        brick(16, 8); question(17, 8); brick(18, 8);
        place_enemy(15.0);   // first goomba — early warning

        // ======== Gap gauntlet ========
        struct Section { int ground_len; bool has_pipe; bool has_blocks; bool has_enemy; };
        Section sections[] = {
            { 7, false, false, true },   //  1
            { 9, true,  false, true },   //  2: pipe + goomba
            { 6, false, true,  false},   //  3: blocks
            { 8, true,  false, true },   //  4: pipe + goomba
            {10, false, false, true },   //  5: open + goomba
            { 5, false, true,  false},   //  6: tight
            { 7, true,  false, true },   //  7: pipe + goomba
            { 9, false, false, true },   //  8: open + goomba
            { 6, true,  true,  false},   //  9: pipe + blocks
            { 8, false, false, true },   // 10: open + goomba
            { 7, false, true,  true },   // 11: blocks + goomba
            { 5, true,  false, false},   // 12: tight + pipe
            { 9, false, false, true },   // 13: open + goomba
            { 8, true,  false, true },   // 14: pipe + goomba
            { 6, false, false, false},   // 15: final approach
        };

        int x = 25;
        for (int i = 0; i < 15; ++i) {
            // 3-tile gap (x, x+1, x+2 stay AIR)
            x += 3;

            int g0 = x;
            int g1 = x + sections[i].ground_len - 1;
            ground(g0, g1);

            if (sections[i].has_pipe) {
                int pm = g0 + sections[i].ground_len / 2;
                pipe(pm, 2);
            }
            if (sections[i].has_blocks) {
                int bm = g0 + 2;
                question(bm, 8);
                if (bm + 2 <= g1) brick(bm + 2, 8);
            }
            if (sections[i].has_enemy) {
                // Place goomba roughly in the middle of the ground section
                double ex = g0 + sections[i].ground_len * 0.5;
                place_enemy(ex);
            }

            x = g1 + 1;
        }

        // ======== Final staircase + flag ========
        x += 3;
        ground(x, x + 3);
        int stair_start = x + 4;
        for (int i = 0; i < 4; ++i)
            for (int h = 0; h <= i; ++h)
                tiles_[11 - h][stair_start + i] = Tile::GROUND;
        ground(stair_start, stair_start + 3);

        flag_x_ = stair_start + 6;
        ground(stair_start + 4, flag_x_ + 2);
        for (int r = 2; r <= 11; ++r)
            tiles_[r][flag_x_] = Tile::FLAG;

        ground(flag_x_ + 3, LEVEL_W - 1);
    }

    // ------------------------------------------------------------------ state
    PlatformerParams p_;
    std::vector<std::vector<Tile>> tiles_;
    std::vector<Enemy> enemies_;
    std::vector<Enemy> initial_enemies_;

    static constexpr int ground_row_ = 12;

    double x_  = 0.0, y_  = 0.0;
    double vx_ = 0.0, vy_ = 0.0;
    double max_x_ = 0.0;
    bool   on_ground_    = true;
    bool   reached_flag_ = false;
    bool   dead_         = false;
    int    step_         = 0;
    int    idle_         = 0;
    int    flag_x_       = 198;
};

} // namespace env
