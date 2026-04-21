#include "neat/neat.hpp"
#include "platformer.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// ============================================================================
// Evaluate a single network on the platformer level.
// Fitness = furthest x reached + big bonus for reaching the flag.
// Level is deterministic, so one trial is enough.
// ============================================================================

static constexpr double FLAG_BONUS = 1000.0;

static double evaluate(neat::Network& net) {
    env::Platformer env;
    neat::Random rng(42);
    env.reset(rng);

    while (!env.terminated()) {
        auto out = net.activate(env.observe());
        env.step(out[0], out[1], out[2]);
    }

    double fitness = env.max_x();
    if (env.reached_flag()) fitness += FLAG_BONUS;
    return fitness;
}

// ============================================================================
// Record trajectory + enemy states as JSON for visualization
// ============================================================================

struct RecordedData {
    std::string traj_json;    // [[mx, my], ...]
    std::string enemies_json; // [[[ex,ey,alive], ...], ...]  per-frame
};

static RecordedData record(neat::Network& net) {
    env::Platformer env;
    neat::Random rng(42);
    env.reset(rng);

    std::ostringstream tj, ej;
    tj << "[";
    ej << "[";

    auto write_frame = [&](bool first) {
        if (!first) { tj << ","; ej << ","; }

        tj << "[" << env.x() << "," << env.y() << "]";

        ej << "[";
        const auto& enemies = env.enemies();
        for (size_t i = 0; i < enemies.size(); ++i) {
            if (i > 0) ej << ",";
            ej << "[" << enemies[i].x << "," << enemies[i].y << ","
               << (enemies[i].alive ? 1 : 0) << "]";
        }
        ej << "]";
    };

    write_frame(true);

    while (!env.terminated()) {
        auto out = net.activate(env.observe());
        env.step(out[0], out[1], out[2]);
        write_frame(false);
    }

    tj << "]";
    ej << "]";
    return {tj.str(), ej.str()};
}

// ============================================================================
// Serialize level tiles as JSON 2D array
// ============================================================================

static std::string level_json(const env::Platformer& env) {
    std::ostringstream ss;
    ss << "[";
    for (int r = 0; r < env.level_height(); ++r) {
        if (r > 0) ss << ",";
        ss << "[";
        for (int c = 0; c < env.level_width(); ++c) {
            if (c > 0) ss << ",";
            ss << static_cast<int>(env.tile_at(c, r));
        }
        ss << "]";
    }
    ss << "]";
    return ss.str();
}

// ============================================================================
// Write self-contained HTML visualization
// ============================================================================

static void write_html(const RecordedData& data,
                       const std::string& tiles_json,
                       int level_w, int level_h,
                       double mario_w, double mario_h,
                       const std::string& path)
{
    std::ofstream f(path);
    f << R"html(<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Platformer — NEAT</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #1a1b26; display: flex; flex-direction: column;
         align-items: center; justify-content: center; height: 100vh;
         font-family: monospace; color: #a9b1d6; }
  canvas { border-radius: 8px; }
  .info { margin-top: 16px; font-size: 14px; }
  .controls { margin-top: 12px; display: flex; gap: 12px; align-items: center; }
  button { background: #7aa2f7; color: #1a1b26; border: none; padding: 6px 16px;
           border-radius: 4px; cursor: pointer; font-family: monospace; font-size: 13px; }
  button:hover { background: #89b4fa; }
  input[type=range] { width: 120px; }
</style>
</head>
<body>
<canvas id="c" width="960" height="560"></canvas>
<div class="info"><span id="status">Frame 0</span></div>
<div class="controls">
  <button id="playBtn">Play</button>
  <button id="resetBtn">Reset</button>
  <label>Speed <input type="range" id="speed" min="1" max="10" value="3"></label>
</div>
<script>
const TRAJ = )html";

    f << data.traj_json;

    f << R"html(;
const ENEMIES = )html";

    f << data.enemies_json;

    f << R"html(;
const TILES = )html";

    f << tiles_json;

    f << R"html(;
const LW = )html" << level_w << R"html(;
const LH = )html" << level_h << R"html(;
const MW = )html" << mario_w << R"html(;
const MH = )html" << mario_h << R"html(;
const EW = )html" << env::Enemy::W << R"html(;
const EH = )html" << env::Enemy::H << R"html(;

const canvas = document.getElementById('c');
const ctx    = canvas.getContext('2d');
const status = document.getElementById('status');
const playBtn    = document.getElementById('playBtn');
const resetBtn   = document.getElementById('resetBtn');
const speedSlider = document.getElementById('speed');

const W = canvas.width, H = canvas.height;
const TILE = Math.floor(H / LH);

// Tile colours  (index matches Tile enum)
const TCOL = [
    null,           // AIR
    '#8B6914',      // GROUND
    '#A0522D',      // BRICK
    '#FFD700',      // QUESTION
    '#228B22',      // PIPE
    '#c0caf5',      // FLAG
];

let frame = 0, playing = false, animId = null;

function draw() {
    ctx.fillStyle = '#6495ED';
    ctx.fillRect(0, 0, W, H);

    const [mx, my] = TRAJ[frame];
    const camX = mx - (W / TILE) / 2 + MW / 2;

    // --- tiles ---
    const startCol = Math.max(0, Math.floor(camX));
    const endCol   = Math.min(LW, Math.ceil(camX + W / TILE) + 1);
    for (let r = 0; r < LH; ++r) {
        for (let c = startCol; c < endCol; ++c) {
            const t = TILES[r][c];
            if (t === 0) continue;
            const sx = (c - camX) * TILE;
            const sy = r * TILE;
            ctx.fillStyle = TCOL[t];
            ctx.fillRect(sx, sy, TILE, TILE);

            if (t === 1 || t === 2) {
                ctx.strokeStyle = 'rgba(0,0,0,0.15)';
                ctx.lineWidth = 1;
                ctx.strokeRect(sx, sy, TILE, TILE);
            }
            if (t === 3) {
                ctx.fillStyle = '#1a1b26';
                ctx.font = `bold ${TILE*0.6}px monospace`;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText('?', sx + TILE/2, sy + TILE/2);
            }
            if (t === 4) {
                ctx.fillStyle = 'rgba(255,255,255,0.12)';
                ctx.fillRect(sx, sy, TILE * 0.25, TILE);
            }
            if (t === 5) {
                ctx.fillStyle = '#c0caf5';
                ctx.fillRect(sx + TILE*0.4, sy, TILE*0.2, TILE);
                if (r === 2) {
                    ctx.fillStyle = '#f7768e';
                    ctx.beginPath();
                    ctx.moveTo(sx + TILE*0.6, sy);
                    ctx.lineTo(sx + TILE*0.6 + TILE, sy + TILE*0.4);
                    ctx.lineTo(sx + TILE*0.6, sy + TILE*0.8);
                    ctx.fill();
                }
            }
        }
    }

    // --- enemies ---
    const enemies = ENEMIES[frame];
    for (const e of enemies) {
        if (!e[2]) continue;   // dead
        const ex = (e[0] - camX) * TILE;
        const ey = e[1] * TILE;
        // body
        ctx.fillStyle = '#b4637a';
        ctx.fillRect(ex, ey, EW * TILE, EH * TILE);
        // feet
        ctx.fillStyle = '#87563d';
        ctx.fillRect(ex, ey + EH * TILE * 0.7, EW * TILE, EH * TILE * 0.3);
        // eyes
        ctx.fillStyle = '#fff';
        ctx.fillRect(ex + EW*TILE*0.15, ey + EH*TILE*0.2, EW*TILE*0.25, EH*TILE*0.15);
        ctx.fillRect(ex + EW*TILE*0.55, ey + EH*TILE*0.2, EW*TILE*0.25, EH*TILE*0.15);
        // angry brow
        ctx.strokeStyle = '#1a1b26';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(ex + EW*TILE*0.1, ey + EH*TILE*0.15);
        ctx.lineTo(ex + EW*TILE*0.45, ey + EH*TILE*0.25);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(ex + EW*TILE*0.9, ey + EH*TILE*0.15);
        ctx.lineTo(ex + EW*TILE*0.55, ey + EH*TILE*0.25);
        ctx.stroke();
    }

    // --- Mario ---
    const msx = (mx - camX) * TILE;
    const msy = my * TILE;
    ctx.fillStyle = '#f7768e';
    ctx.fillRect(msx, msy, MW * TILE, MH * TILE);
    ctx.fillStyle = '#fff';
    ctx.fillRect(msx + MW*TILE*0.55, msy + MH*TILE*0.2, MW*TILE*0.2, MH*TILE*0.15);

    // --- HUD ---
    let maxX = 0;
    for (let i = 0; i <= frame; ++i) maxX = Math.max(maxX, TRAJ[i][0]);
    let alive = 0;
    for (const e of enemies) if (e[2]) alive++;
    status.textContent =
        `Frame ${frame}/${TRAJ.length-1}  |  x: ${mx.toFixed(1)}  |  ` +
        `max: ${maxX.toFixed(1)}  |  enemies: ${alive}`;
}

function tick() {
    if (!playing) return;
    frame = Math.min(frame + parseInt(speedSlider.value), TRAJ.length - 1);
    draw();
    if (frame >= TRAJ.length - 1) { playing = false; playBtn.textContent = 'Play'; }
    else animId = requestAnimationFrame(tick);
}

playBtn.addEventListener('click', () => {
    if (playing) {
        playing = false; playBtn.textContent = 'Play'; cancelAnimationFrame(animId);
    } else {
        if (frame >= TRAJ.length - 1) { frame = 0; }
        playing = true; playBtn.textContent = 'Pause'; tick();
    }
});

resetBtn.addEventListener('click', () => {
    playing = false; playBtn.textContent = 'Play';
    cancelAnimationFrame(animId);
    frame = 0; draw();
});

draw();
</script>
</body>
</html>
)html";
    f.close();
}

// ============================================================================
// Main
// ============================================================================

int main() {
    neat::Config cfg;
    cfg.num_inputs  = env::Platformer::NUM_OBS;   // 20
    cfg.num_outputs = 3;   // left, right, jump (SIGMOID, thresholded at 0.5)

    cfg.population_size    = 300;
    cfg.seed               = 421283497;
    cfg.parallel_eval      = true;

    cfg.compat_threshold   = 1.0;
    cfg.c1 = 1.0; cfg.c2 = 1.0; cfg.c3 = 0.0;
    cfg.dropoff_age        = 20;

    cfg.prob_mutate_weight    = 0.8;
    cfg.prob_weight_replaced  = 0.1;
    cfg.weight_mutation_power = 0.5;

    // Higher chance to add nodes/links than in typical NEAT, since Mario likely needs more complex topology than XOR.
    // good values are: add_node ~0.05-0.1, add_link ~0.06-0.08, toggle_enable ~0.01-0.05
    cfg.prob_add_node         = 0.06;
    cfg.prob_add_link         = 0.07;
    cfg.prob_toggle_enable    = 0.05;

    cfg.prob_crossover        = 0.75;
    cfg.prob_reenable_gene    = 0.25;
    cfg.survival_threshold    = 0.2;

    cfg.activation = neat::ActivationType::SIGMOID;

    neat::Population pop(cfg);

    constexpr int    MAX_GENS      = 1000;
    constexpr double SOLVED_THRESH = 1100.0;

    std::printf("Platformer (SMB 1-1) — NEAT\n");
    std::printf("Inputs: %u  Outputs: %u  Population: %u\n\n",
        cfg.num_inputs, cfg.num_outputs, cfg.population_size);
    std::printf("gen | best     | mean     | worst    | species\n");
    std::printf("----|----------|----------|----------|--------\n");

    pop.run_until(
        [](neat::Network& net) { return evaluate(net); },
        [](const neat::GenerationResult& r) {
            std::printf("%3u | %8.2f | %8.2f | %8.2f | %3u\n",
                r.generation, r.best_fitness,
                r.mean_fitness, r.worst_fitness,
                r.num_species);
            if (r.best_fitness >= SOLVED_THRESH) {
                std::printf("\nSolved at generation %u!\n", r.generation);
                return true;
            }
            return r.generation >= MAX_GENS;
        }
    );

    std::printf("\nRecording best network trajectory...\n");
    auto best = pop.best_network();
    auto data = record(best);

    env::Platformer env;
    neat::Random rng(42);
    env.reset(rng);
    auto tiles = level_json(env);

    env::PlatformerParams params;
    write_html(data, tiles,
               env::Platformer::LEVEL_W, env::Platformer::LEVEL_H,
               params.mario_w, params.mario_h,
               "platformer_viz.html");
    std::printf("Wrote platformer_viz.html — open in browser.\n");

    return 0;
}
