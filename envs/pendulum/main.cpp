#include "neat/neat.hpp"
#include "double_pendulum.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>


// ============================================================================
// Evaluate a single network on multiple episodes.
// Each trial uses a fixed seed so every genome in the same generation
// faces identical initial conditions — fitness comparisons are fair.
// ============================================================================

static constexpr int    NUM_TRIALS      = 3;
static constexpr double UPRIGHT_BONUS   = 100.0; // one-time reward for first reaching upright
static constexpr double STABILITY_SCALE = 5.0;   // per-step multiplier while holding upright
static constexpr double CALM_SCALE     = 2.0;   // per-step reward for link 2 being calm (scaled by link 1 height)

static double evaluate(neat::Network& net) {
    env::DoublePendulum env;
    double total = 0.0;

    for (int trial = 0; trial < NUM_TRIALS; ++trial) {
        neat::Random rng(static_cast<uint64_t>(trial) + 7777);
        env.reset(rng);

        double episode_reward = 0.0;
        bool   bonus_given    = false;
        while (!env.terminated()) {
            auto out = net.activate(env.observe());
            env.step(out[0], out[1]);

            // Continuous height signal — gradient everywhere, even when swinging.
            double height = (-std::cos(env.theta1()) - std::cos(env.theta2()) + 2.0) / 4.0;

            double stability = 0.0;
            if (env.is_upright()) {
                // One-time bonus for first reaching the upright window.
                if (!bonus_given) {
                    episode_reward += UPRIGHT_BONUS;
                    bonus_given = true;
                }
                // Per-step reward for holding — penalised by velocity.
                double v1 = env.theta1_dot() / env.max_vel();
                double v2 = env.theta2_dot() / env.max_vel();
                stability = 1.0 - std::min(1.0, v1 * v1 + v2 * v2);
            }

            // Reward link 2 being calm, but only when link 1 is high.
            // Avoids fighting the swing-up phase (link 1 low → no penalty).
            double link1_up = (-std::cos(env.theta1()) + 1.0) / 2.0; // [0,1]
            double v2_norm  = env.theta2_dot() / env.max_vel();
            double link2_calm = 1.0 - std::min(1.0, v2_norm * v2_norm);
            double calm = link1_up * link2_calm; // only matters when link 1 is up

            episode_reward += height + STABILITY_SCALE * stability + CALM_SCALE * calm;
        }

        total += episode_reward;
    }

    return total / NUM_TRIALS;
}

// ============================================================================
// Record a trajectory as JSON for visualization
// ============================================================================

struct Frame { double theta1, theta2; };

static std::string record_trajectory(neat::Network& net, uint64_t seed) {
    env::DoublePendulum env;
    neat::Random rng(seed);
    env.reset(rng);

    std::vector<Frame> frames;
    frames.push_back({env.theta1(), env.theta2()});

    while (!env.terminated()) {
        auto out = net.activate(env.observe());
        env.step(out[0], out[1]);
        frames.push_back({env.theta1(), env.theta2()});
    }

    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < frames.size(); ++i) {
        if (i > 0) ss << ",";
        ss << "[" << frames[i].theta1 << "," << frames[i].theta2 << "]";
    }
    ss << "]";
    return ss.str();
}

// ============================================================================
// Write self-contained HTML visualization
// ============================================================================

static void write_html(const std::string& traj_json,
                       double l1, double l2,
                       const std::string& path)
{
    std::ofstream f(path);
    f << R"html(<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Double Pendulum — NEAT</title>
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
<canvas id="c" width="600" height="600"></canvas>
<div class="info"><span id="status">Frame 0</span></div>
<div class="controls">
  <button id="playBtn">Play</button>
  <button id="resetBtn">Reset</button>
  <label>Speed <input type="range" id="speed" min="1" max="10" value="3"></label>
</div>
<script>
const DATA = )html";

    f << traj_json;

    f << R"html(;
const L1 = )html" << l1 << R"html(;
const L2 = )html" << l2 << R"html(;

const canvas = document.getElementById('c');
const ctx    = canvas.getContext('2d');
const status = document.getElementById('status');
const playBtn    = document.getElementById('playBtn');
const resetBtn   = document.getElementById('resetBtn');
const speedSlider = document.getElementById('speed');

const W = canvas.width, H = canvas.height;
const cx = W / 2, cy = H / 2;
const scale = Math.min(W, H) * 0.35 / (L1 + L2);

let frame = 0, playing = false, animId = null;

// Trail buffer
const TRAIL_LEN = 40;
const trail = [];

function draw() {
    ctx.fillStyle = '#1a1b26';
    ctx.fillRect(0, 0, W, H);

    const [t1, t2] = DATA[frame];

    // Pivot is at centre; angles from downward vertical.
    // Screen y increases downward, so y-offset = cos(θ) * L * scale.
    const px1 = cx + Math.sin(t1) * L1 * scale;
    const py1 = cy + Math.cos(t1) * L1 * scale;
    const px2 = px1 + Math.sin(t2) * L2 * scale;
    const py2 = py1 + Math.cos(t2) * L2 * scale;

    // Update tip trail
    trail.push([px2, py2]);
    if (trail.length > TRAIL_LEN) trail.shift();

    // Upright guide (dashed)
    ctx.strokeStyle = '#2a2b3a';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 6]);
    ctx.beginPath();
    ctx.moveTo(cx, cy - (L1 + L2) * scale - 20);
    ctx.lineTo(cx, cy);
    ctx.stroke();
    ctx.setLineDash([]);

    // Tip trail
    for (let i = 1; i < trail.length; ++i) {
        const alpha = i / trail.length;
        ctx.strokeStyle = `rgba(247,118,142,${alpha * 0.6})`;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(trail[i-1][0], trail[i-1][1]);
        ctx.lineTo(trail[i][0],   trail[i][1]);
        ctx.stroke();
    }

    // Link 1
    ctx.strokeStyle = '#7aa2f7';
    ctx.lineWidth = 5;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(px1, py1);
    ctx.stroke();

    // Link 2
    ctx.strokeStyle = '#e0af68';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(px1, py1);
    ctx.lineTo(px2, py2);
    ctx.stroke();

    // Joints
    for (const [jx, jy, r, col] of [
            [cx,  cy,  8, '#565f89'],
            [px1, py1, 5, '#c0caf5'],
            [px2, py2, 4, '#f7768e']]) {
        ctx.fillStyle = col;
        ctx.beginPath();
        ctx.arc(jx, jy, r, 0, Math.PI * 2);
        ctx.fill();
    }

    const deg = v => (v * 180 / Math.PI).toFixed(1);
    status.textContent =
        `Frame ${frame} / ${DATA.length-1}  |  θ₁: ${deg(t1)}°  θ₂: ${deg(t2)}°`;
}

function tick() {
    if (!playing) return;
    frame = Math.min(frame + parseInt(speedSlider.value), DATA.length - 1);
    draw();
    if (frame >= DATA.length - 1) { playing = false; playBtn.textContent = 'Play'; }
    else animId = requestAnimationFrame(tick);
}

playBtn.addEventListener('click', () => {
    if (playing) {
        playing = false; playBtn.textContent = 'Play'; cancelAnimationFrame(animId);
    } else {
        if (frame >= DATA.length - 1) { frame = 0; trail.length = 0; }
        playing = true; playBtn.textContent = 'Pause'; tick();
    }
});

resetBtn.addEventListener('click', () => {
    playing = false; playBtn.textContent = 'Play';
    cancelAnimationFrame(animId);
    frame = 0; trail.length = 0; draw();
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

int main(int argc, char* argv[]) {
    std::string csv_path = neat::parse_csv_arg(argc, argv, "pendulum_results.csv");
    neat::Config cfg;
    cfg.num_inputs  = 6;   // cos(θ1), sin(θ1), cos(θ2), sin(θ2), θ1_dot, θ2_dot
    cfg.num_outputs = 2;   // torque at joint 1 and joint 2

    cfg.population_size    = 150;
    cfg.seed               = 31415926;
    cfg.parallel_eval      = true;

    cfg.compat_threshold   = 3.0;
    // set c3 = 0.0 to ignore weight differences and focus on topology speciation,
    cfg.c1 = 1.0; cfg.c2 = 1.0; cfg.c3 = 0.4;
    cfg.dropoff_age        = 20;

    cfg.prob_mutate_weight    = 0.8;
    cfg.prob_weight_replaced  = 0.1;
    cfg.weight_mutation_power = 0.5;

    // set 3 below to 0.0 to disable structural mutations and focus on weight evolution,
    // default values are: 0.03 for add node, 0.05 for add link, 0.01 for toggle enable.
    cfg.prob_add_node         = 0.03;
    cfg.prob_add_link         = 0.05;
    cfg.prob_toggle_enable    = 0.01;

    cfg.prob_crossover        = 0.75;
    cfg.prob_reenable_gene    = 0.25;
    cfg.survival_threshold    = 0.2;

    // TANH gives a symmetric output centred on 0 before the [0,1] mapping,
    // which suits continuous torque control better than SIGMOID.
    cfg.activation = neat::ActivationType::TANH;

    neat::parse_config_args(cfg, argc, argv);

    constexpr int    MAX_GENS      = 1000;
    constexpr double SOLVED_THRESH = 15700.0; // ~80% of max per trial (6100), averaged over 5 trials

    if (!csv_path.empty()) {
        neat::write_config(neat::config_sidecar_path(csv_path), "pendulum", cfg, {
            {"max_generations", std::to_string(MAX_GENS)},
            {"solved_threshold", std::to_string(SOLVED_THRESH)},
            {"num_trials",      std::to_string(NUM_TRIALS)},
            {"upright_bonus",   std::to_string(UPRIGHT_BONUS)},
            {"stability_scale", std::to_string(STABILITY_SCALE)},
            {"calm_scale",      std::to_string(CALM_SCALE)},
        });
    }

    neat::Population pop(cfg);

    std::printf("Double Pendulum Balance — NEAT\n");
    std::printf("Inputs: %u  Outputs: %u  Population: %u\n\n",
        cfg.num_inputs, cfg.num_outputs, cfg.population_size);
    std::printf("gen | best     | mean     | worst    | species\n");
    std::printf("----|----------|----------|----------|--------\n");

    auto run = pop.run_until(
        [](neat::Network& net) { return evaluate(net); },
        [](const neat::GenerationResult& r) {
            std::printf("%3u | %8.4f | %8.4f | %8.4f | %3u\n",
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

    if (!csv_path.empty()) {
        neat::write_csv(csv_path, run.generations, cfg.seed);
        std::printf("Wrote %s\n", csv_path.c_str());
    }

    std::printf("\nRecording best network trajectory...\n");
    auto best = pop.best_network();
    auto traj = record_trajectory(best, 42);

    env::DoublePendulumParams params;
    write_html(traj, params.l1, params.l2, "pendulum_viz.html");
    std::printf("Wrote pendulum_viz.html — open in browser.\n");

    return 0;
}
