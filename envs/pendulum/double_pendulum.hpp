#pragma once

#include "neat/random.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <vector>

namespace env {

struct DoublePendulumParams {
    double m1          = 1.0;    // mass of link 1 (kg)
    double m2          = 1.0;    // mass of link 2 (kg)
    double l1          = 1.0;    // length of link 1 (m)
    double l2          = 1.0;    // length of link 2 (m)
    double gravity     = 9.81;   // m/s²
    double dt          = 0.01;   // simulation timestep (s)
    double max_torque  = 22.0;   // max torque at joint 1 (N⋅m) — gravity torque at horizontal ≈ 19.6 N⋅m
    double max_vel     = 8.0;    // velocity clamp for normalisation
    double angle_limit = 1.0;    // radians from upright before termination (~57°)
    int    max_steps   = 2000;
    double init_noise  = 0.1;    // initial angle noise (radians from upright)
};

// ============================================================================
// DoublePendulum — balance task
//
// Both links start near upright. The network applies torque at joint 1 only
// (joint 2 is passive), making this an underactuated control problem.
//
// Angles θ1, θ2 are measured from the DOWNWARD vertical.
// Upright position: θ1 = θ2 = π.
//
// Observations (6): cos(θ1), sin(θ1), cos(θ2), sin(θ2),
//                   θ1_dot (normalised), θ2_dot (normalised)
// Action (1):       torque fraction ∈ [0,1] → scaled to [-max_torque, max_torque]
// ============================================================================

class DoublePendulum {
public:
    explicit DoublePendulum(const DoublePendulumParams& p = {}) : p_(p) {}

    void reset(neat::Random& rng) {
        auto noise = [&](double range) {
            return (rng.random_double() * 2.0 - 1.0) * range;
        };
        // Start near downward position (θ = 0). The network must swing up.
        theta1_         = noise(p_.init_noise);
        theta2_         = noise(p_.init_noise);
        theta1_dot_     = noise(0.02);
        theta2_dot_     = noise(0.02);
        step_           = 0;
        reached_upright_ = false;
    }

    // actions ∈ (-1,1) from TANH network outputs, mapped to [-max_torque, max_torque]
    void step(double action1, double action2 = 0.0) {
        double tau1 = std::clamp(action1, -1.0, 1.0) * p_.max_torque;
        double tau2 = std::clamp(action2, -1.0, 1.0) * p_.max_torque;

        double s1  = std::sin(theta1_);
        double s2  = std::sin(theta2_);
        double s12 = std::sin(theta1_ - theta2_);
        double c12 = std::cos(theta1_ - theta2_);

        // Inertia matrix entries (Lagrangian mechanics, point masses at link ends)
        double M11 = (p_.m1 + p_.m2) * p_.l1 * p_.l1;
        double M12 = p_.m2 * p_.l1 * p_.l2 * c12;
        double M22 = p_.m2 * p_.l2 * p_.l2;

        // Coriolis coupling term
        double h = p_.m2 * p_.l1 * p_.l2 * s12;

        // Gravity torques (angles from downward, so gravity pulls toward 0)
        double G1 = -(p_.m1 + p_.m2) * p_.gravity * p_.l1 * s1;
        double G2 = -p_.m2 * p_.gravity * p_.l2 * s2;

        // RHS of  M * [θ1_ddot, θ2_ddot]ᵀ = rhs
        double rhs1 = G1 + h * theta2_dot_ * theta2_dot_ + tau1;
        double rhs2 = G2 - h * theta1_dot_ * theta1_dot_ + tau2;

        // Solve 2×2 linear system via Cramer's rule
        double det        = M11 * M22 - M12 * M12;
        double theta1_ddot = (M22 * rhs1 - M12 * rhs2) / det;
        double theta2_ddot = (M11 * rhs2 - M12 * rhs1) / det;

        // Semi-implicit Euler integration
        theta1_dot_ += p_.dt * theta1_ddot;
        theta2_dot_ += p_.dt * theta2_ddot;

        // Clamp velocities to prevent numerical blow-up from large torques.
        const double vel_limit = p_.max_vel * 5.0;
        theta1_dot_ = std::clamp(theta1_dot_, -vel_limit, vel_limit);
        theta2_dot_ = std::clamp(theta2_dot_, -vel_limit, vel_limit);

        theta1_ += p_.dt * theta1_dot_;
        theta2_ += p_.dt * theta2_dot_;

        if (is_upright()) reached_upright_ = true;
        ++step_;
    }

    std::vector<double> observe() const {
        return {
            std::cos(theta1_),
            std::sin(theta1_),
            std::cos(theta2_),
            std::sin(theta2_),
            std::clamp(theta1_dot_ / p_.max_vel, -1.0, 1.0),
            std::clamp(theta2_dot_ / p_.max_vel, -1.0, 1.0),
        };
    }

    bool terminated() const {
        return step_ >= p_.max_steps
            || std::isnan(theta1_) || std::isnan(theta2_);
    }

    // Both links within angle_limit radians of upright (θ = π).
    // cos(θ) ≈ -1 at upright; cos(θ) < -cos(angle_limit) means close enough.
    bool is_upright() const {
        double threshold = -std::cos(p_.angle_limit);
        return std::cos(theta1_) < threshold && std::cos(theta2_) < threshold;
    }

    int    step_count()  const { return step_; }
    int    max_steps()   const { return p_.max_steps; }
    double theta1()      const { return theta1_; }
    double theta2()      const { return theta2_; }
    double theta1_dot()  const { return theta1_dot_; }
    double theta2_dot()  const { return theta2_dot_; }
    double max_vel()     const { return p_.max_vel; }

private:
    DoublePendulumParams p_;
    double theta1_     = std::numbers::pi;
    double theta2_     = std::numbers::pi;
    double theta1_dot_     = 0.0;
    double theta2_dot_     = 0.0;
    int    step_           = 0;
    bool   reached_upright_ = false;
};

} // namespace env
