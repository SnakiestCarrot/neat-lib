#include "neat/population.hpp"
#include "neat/network.hpp"
#include "neat/config.hpp"
#include "cartpole.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// ============================================================================
// Evaluate a single network on multiple cart-pole episodes
// ============================================================================

static constexpr int NUM_TRIALS = 5;

static double evaluate(neat::Network& net) {
    env::CartPole env;
    double total_fitness = 0.0;

    for (int trial = 0; trial < NUM_TRIALS; ++trial) {
        // Fresh RNG per trial with a fixed seed — every genome in the
        // generation faces the same set of initial conditions.
        neat::Random trial_rng(static_cast<uint64_t>(trial) + 1000);
        env.reset(trial_rng);

        while (!env.terminated()) {
            auto obs = env.observe();
            auto out = net.activate(obs);
            double action = out[0] * 2.0 - 1.0;
            env.step(action);
        }

        double survival = static_cast<double>(env.step_count())
                        / static_cast<double>(env.max_steps());
        total_fitness += survival;
    }

    return total_fitness / NUM_TRIALS;
}

// ============================================================================
// Record a trajectory as JSON for visualization
// ============================================================================

struct Frame {
    double x, theta;
};

static std::string record_trajectory(neat::Network& net, uint64_t seed) {
    env::CartPole env;
    neat::Random rng(seed);
    env.reset(rng);

    std::vector<Frame> frames;
    frames.push_back({env.x(), env.theta()});

    while (!env.terminated()) {
        auto obs = env.observe();
        auto out = net.activate(obs);
        double action = out[0] * 2.0 - 1.0;
        env.step(action);
        frames.push_back({env.x(), env.theta()});
    }

    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < frames.size(); ++i) {
        if (i > 0) ss << ",";
        ss << "[" << frames[i].x << "," << frames[i].theta << "]";
    }
    ss << "]";
    return ss.str();
}

// ============================================================================
// Write self-contained HTML visualization
// ============================================================================

static void write_html(const std::string& trajectory_json,
                       double track_limit, double pole_len,
                       const std::string& path)
{
    std::ofstream f(path);
    f << R"html(<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>CartPole — NEAT</title>
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
<canvas id="c" width="800" height="400"></canvas>
<div class="info">
  <span id="status">Frame 0</span>
</div>
<div class="controls">
  <button id="playBtn">Play</button>
  <button id="resetBtn">Reset</button>
  <label>Speed <input type="range" id="speed" min="1" max="10" value="3"></label>
</div>
<script>
const DATA = )html";

    f << trajectory_json;

    f << R"html(;
const TRACK = )html" << track_limit << R"html(;
const POLE_LEN = )html" << pole_len << R"html(;

const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const status = document.getElementById('status');
const playBtn = document.getElementById('playBtn');
const resetBtn = document.getElementById('resetBtn');
const speedSlider = document.getElementById('speed');

let frame = 0;
let playing = false;
let animId = null;

const W = canvas.width, H = canvas.height;
const groundY = H * 0.7;
const scale = (W * 0.4) / TRACK;

function toScreen(x) { return W / 2 + x * scale; }

function draw() {
    ctx.fillStyle = '#1a1b26';
    ctx.fillRect(0, 0, W, H);

    const [x, theta] = DATA[frame];

    // Track
    ctx.strokeStyle = '#565f89';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(toScreen(-TRACK), groundY);
    ctx.lineTo(toScreen(TRACK), groundY);
    ctx.stroke();

    // Track limits
    ctx.strokeStyle = '#f7768e';
    ctx.setLineDash([4, 4]);
    for (const lim of [-TRACK, TRACK]) {
        ctx.beginPath();
        ctx.moveTo(toScreen(lim), groundY - 20);
        ctx.lineTo(toScreen(lim), groundY + 20);
        ctx.stroke();
    }
    ctx.setLineDash([]);

    // Cart
    const cartW = 40, cartH = 20;
    const cx = toScreen(x);
    ctx.fillStyle = '#7aa2f7';
    ctx.fillRect(cx - cartW/2, groundY - cartH/2, cartW, cartH);

    // Wheels
    ctx.fillStyle = '#565f89';
    for (const dx of [-12, 12]) {
        ctx.beginPath();
        ctx.arc(cx + dx, groundY + cartH/2 + 4, 4, 0, Math.PI * 2);
        ctx.fill();
    }

    // Pole
    const polePixels = POLE_LEN * 2 * scale;
    const px = cx + Math.sin(theta) * polePixels;
    const py = groundY - Math.cos(theta) * polePixels;

    ctx.strokeStyle = '#e0af68';
    ctx.lineWidth = 4;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(cx, groundY);
    ctx.lineTo(px, py);
    ctx.stroke();

    // Pole tip
    ctx.fillStyle = '#e0af68';
    ctx.beginPath();
    ctx.arc(px, py, 5, 0, Math.PI * 2);
    ctx.fill();

    // Hinge
    ctx.fillStyle = '#c0caf5';
    ctx.beginPath();
    ctx.arc(cx, groundY, 4, 0, Math.PI * 2);
    ctx.fill();

    status.textContent = `Frame ${frame} / ${DATA.length - 1}  |  x: ${x.toFixed(3)} m  |  \u03B8: ${(theta * 180 / Math.PI).toFixed(1)}\u00B0`;
}

function tick() {
    if (!playing) return;
    const spd = parseInt(speedSlider.value);
    frame = Math.min(frame + spd, DATA.length - 1);
    draw();
    if (frame >= DATA.length - 1) {
        playing = false;
        playBtn.textContent = 'Play';
    } else {
        animId = requestAnimationFrame(tick);
    }
}

playBtn.addEventListener('click', () => {
    if (playing) {
        playing = false;
        playBtn.textContent = 'Play';
        cancelAnimationFrame(animId);
    } else {
        if (frame >= DATA.length - 1) frame = 0;
        playing = true;
        playBtn.textContent = 'Pause';
        tick();
    }
});

resetBtn.addEventListener('click', () => {
    playing = false;
    playBtn.textContent = 'Play';
    cancelAnimationFrame(animId);
    frame = 0;
    draw();
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
    cfg.num_inputs  = 4;
    cfg.num_outputs = 1;

    cfg.population_size    = 150;
    cfg.seed               = 42;
    cfg.parallel_eval      = true;

    cfg.compat_threshold   = 3.0;
    cfg.c1 = 1.0; cfg.c2 = 1.0; cfg.c3 = 0.4;
    cfg.dropoff_age        = 15;

    cfg.prob_mutate_weight   = 0.8;
    cfg.prob_weight_replaced = 0.1;
    cfg.weight_mutation_power = 0.5;

    // set 3 below to 0.0 to disable structural mutations and just evolve weights on the fixed topology
    cfg.prob_add_node        = 0.03;
    cfg.prob_add_link        = 0.05;
    cfg.prob_toggle_enable   = 0.01;

    cfg.prob_crossover       = 0.75;
    cfg.prob_reenable_gene   = 0.25;
    cfg.survival_threshold   = 0.2;
    cfg.c3 = 0.4; // set to 0.0 to ignore weight differences in speciation, 0.4 is a common default

    cfg.activation = neat::ActivationType::SIGMOID;

    neat::Population pop(cfg);

    constexpr int    MAX_GENS      = 300;
    constexpr double SOLVED_THRESH = 0.95;

    std::printf("gen | best     | mean     | worst    | species\n");
    std::printf("----|----------|----------|----------|--------\n");

    for (int gen = 0; gen < MAX_GENS; ++gen) {
        auto result = pop.run_generation([](neat::Network& net) {
            return evaluate(net);
        });

        std::printf("%3u | %8.4f | %8.4f | %8.4f | %3u\n",
            result.generation, result.best_fitness,
            result.mean_fitness, result.worst_fitness,
            result.num_species);

        if (result.best_fitness >= SOLVED_THRESH) {
            std::printf("\nSolved at generation %u!\n", result.generation);
            break;
        }
    }

    std::printf("\nRecording best network trajectory...\n");
    auto best = pop.best_network();
    auto traj = record_trajectory(best, 1000);

    env::CartPoleParams params;
    write_html(traj, params.track_limit, params.pole_half_len, "cartpole_viz.html");
    std::printf("Wrote cartpole_viz.html — open in browser.\n");

    return 0;
}
