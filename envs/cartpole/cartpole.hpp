#pragma once

#include "neat/random.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace env {

// ============================================================================
// CartPole — classic balance task
//
// Observations (4):  x, x_dot, theta, theta_dot  (normalised)
// Action (1):        force direction ∈ [-1, 1]
// ============================================================================

struct CartPoleParams {
    double gravity        = 9.81;
    double cart_mass      = 1.0;
    double pole_mass      = 0.1;
    double pole_half_len  = 0.5;
    double force_mag      = 10.0;
    double dt             = 0.02;

    double track_limit    = 2.4;
    double angle_limit    = 1.57;
    int    max_steps      = 2000;

    double init_angle_range = 1.4;
    double init_vel_range   = 0.05;
};

class CartPole {
public:
    explicit CartPole(const CartPoleParams& params = {})
        : p_(params) {}

    void reset(neat::Random& rng) {
        auto uniform = [&](double range) -> double {
            return (rng.random_double() * 2.0 - 1.0) * range;
        };

        x_         = uniform(0.5 * p_.track_limit);
        x_dot_     = uniform(p_.init_vel_range);
        theta_     = uniform(p_.init_angle_range);
        theta_dot_ = uniform(p_.init_vel_range);
        step_      = 0;
    }

    bool step(double action) {
        double force = std::clamp(action, -1.0, 1.0) * p_.force_mag;

        double cos_th = std::cos(theta_);
        double sin_th = std::sin(theta_);
        double total_mass = p_.cart_mass + p_.pole_mass;
        double pole_ml = p_.pole_mass * p_.pole_half_len;

        double temp = (force + pole_ml * theta_dot_ * theta_dot_ * sin_th)
                      / total_mass;
        double theta_acc = (p_.gravity * sin_th - cos_th * temp)
                           / (p_.pole_half_len
                              * (4.0 / 3.0 - p_.pole_mass * cos_th * cos_th / total_mass));
        double x_acc = temp - pole_ml * theta_acc * cos_th / total_mass;

        x_         += p_.dt * x_dot_;
        x_dot_     += p_.dt * x_acc;
        theta_     += p_.dt * theta_dot_;
        theta_dot_ += p_.dt * theta_acc;

        ++step_;
        return !terminated();
    }

    std::vector<double> observe() const {
        return {
            x_ / p_.track_limit,
            std::clamp(x_dot_ / 3.0, -1.0, 1.0),
            theta_ / p_.angle_limit,
            std::clamp(theta_dot_ / 3.0, -1.0, 1.0)
        };
    }

    bool terminated() const {
        return std::abs(x_) > p_.track_limit
            || std::abs(theta_) > p_.angle_limit
            || step_ >= p_.max_steps;
    }

    bool failed() const {
        return std::abs(x_) > p_.track_limit
            || std::abs(theta_) > p_.angle_limit;
    }

    int    step_count() const { return step_; }
    int    max_steps()  const { return p_.max_steps; }
    double x()         const { return x_; }
    double x_dot()     const { return x_dot_; }
    double theta()     const { return theta_; }
    double theta_dot() const { return theta_dot_; }

    double track_limit()   const { return p_.track_limit; }
    double pole_half_len() const { return p_.pole_half_len; }

private:
    CartPoleParams p_;
    double x_         = 0.0;
    double x_dot_     = 0.0;
    double theta_     = 0.0;
    double theta_dot_ = 0.0;
    int    step_      = 0;
};

} // namespace env
