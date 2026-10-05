#include "model/usv_dynamic_model.h"

#include <algorithm>
#include <cmath>
#include <tuple>

UsvDynamicModel::UsvDynamicModel(const Eigen::Vector3d &pose,
                                 const Eigen::Vector3d &vel,
                                 const ModelParams &params)
    : p(params) {
  eta = pose;
  nu = vel;
  C = Eigen::Matrix3d::Zero();
  D = Eigen::Matrix3d::Zero();

  M << p.m - p.X_u_dot, 0, 0,                      //
      0, p.m - p.Y_v_dot, p.m * p.xg - p.Y_r_dot,  //
      0, p.m * p.xg - p.N_v_dot, p.Iz - p.N_r_dot; //
  M_inv = M.inverse();
}

State UsvDynamicModel::update(double T_port, double T_stbd) {
  double Tp = std::clamp(T_port, T_min, T_max);
  double Ts = std::clamp(T_stbd, T_min, T_max);
  return update(Eigen::Vector3d{Tp + Ts, 0.0, 0.5 * B * (Tp - Ts)});
}

State UsvDynamicModel::update(Eigen::Vector3d u) {
  return update_with_perturb(u, Eigen::Vector3d::Zero());
}

State UsvDynamicModel::update_with_perturb(Eigen::Vector3d F,
                                           const Eigen::Vector3d &nu_c) {
  // Relative velocity vector (nu_r = nu - ocean currents)
  Eigen::Vector3d nu_r = nu - nu_c;
  DecomposedDyn dyn = get_decomposed_dyn(nu);

  // nu_dot = M.inverse() * (F - C * nu - D * nu);
  nu_dot = dyn.f + dyn.g * F;
  nu = p.integral_step * (nu_dot + nu_dot_last) / 2 + nu; // integral
  nu_dot_last = nu_dot;

  nu(0) = std::clamp(nu(0), p.max_astern, p.max_surge);
  nu(2) = std::clamp(nu(2), -p.max_yaw, p.max_yaw);

  Eigen::Matrix3d J;
  J << std::cos(eta(2)), -std::sin(eta(2)), 0, //
      std::sin(eta(2)), std::cos(eta(2)), 0,   //
      0, 0, 1;

  Eigen::Vector3d eta_dot = J * nu; // transformation into local reference frame
  eta = p.integral_step * (eta_dot + eta_dot_last) / 2 + eta; // integral
  eta_dot_last = eta_dot;

  return State{
      eta(0),    eta(1),    eta(2),    //
      nu(0),     nu(1),     nu(2),     //
      nu_dot(0), nu_dot(1), nu_dot(2), //
  };
}

double UsvDynamicModel::wrap_angle(double angle) {
  double wrapped = std::fmod(angle + M_PI, 2 * M_PI);
  if (wrapped < 0)
    wrapped += 2 * M_PI;
  return wrapped - M_PI;
}

std::pair<double, double> UsvDynamicModel::allocate(double X, double N) {
  // X = Tp + Ts, N = 0.5 * B * (Tp - Ts)
  double Tp = X / 2.0 + N / B;
  double Ts = X / 2.0 - N / B;

  // Scale both thrusts equally so the heading intent is kept
  auto limit = [](double T) { return T >= 0 ? T_max / T : T_min / T; };
  double scale = std::min({1.0, limit(Tp), limit(Ts)});
  return {Tp * scale, Ts * scale};
}

DecomposedDyn UsvDynamicModel::get_decomposed_dyn(const Eigen::Vector3d &nu_) {
  DecomposedDyn out{};
  Eigen::Matrix3d C_RB, C_A;
  auto [surge, sway, yaw] = std::make_tuple(nu_.x(), nu_.y(), nu_.z());
  double c0 = p.m * (p.xg * yaw + sway);
  double c1 = p.m * surge;
  double c2 = p.Y_v_dot * sway + 0.5 * (p.Y_r_dot + p.N_v_dot) * yaw;
  double c3 = p.X_u_dot * surge;

  C_RB << 0, 0, -c0, //
      0, 0, c1,      //
      c0, -c1, 0;    //

  C_A << 0, 0, c2, //
      0, 0, -c3,   //
      -c2, c3, 0;  //

  C = C_RB + C_A;

  auto nu_abs = nu_.cwiseAbs();
  auto [surge_abs, sway_abs, yaw_abs] =
      std::make_tuple(nu_abs.x(), nu_abs.y(), nu_abs.z());
  double vel = std::sqrt(surge * surge + sway * sway + 1e-9);

  // Linear damping. Surge drag switches regime above 1.2 m/s.
  double Xu = surge > 1.2 ? 64.55 : -25.0;
  double Xuu = surge > 1.2 ? p.Xuu : 0.0;
  double k = 0.09 * 0.09 * 1.01;
  double Yv = 0.5 * (-40000.0 * sway_abs) *
              (1.1 + 0.0045 * (1.01 / 0.09) - 0.1 * (0.27 / 0.09) +
               0.016 * std::pow(0.27 / 0.09, 2));
  double Yr = 6.0 * (-M_PI * 1000.0) * vel * k;
  double Nv = 0.06 * (-M_PI * 1000.0) * vel * k;
  double Nr = 0.02 * (-M_PI * 1000.0) * vel * k * 1.01;

  D << -Xu - Xuu * surge_abs, 0, 0,                  //
      0, -Yv - (p.Yvv * sway_abs + p.Yvr * yaw_abs), //
      -Yr - (p.Yrv * sway_abs + p.Yrr * yaw_abs),    //
      0, -Nv - (p.Nvv * sway_abs + p.Nvr * yaw_abs), //
      -Nr - (p.Nrv * sway_abs + p.Nrr * yaw_abs);    //

  out.f = -M_inv * (C * nu_ + D * nu_);
  out.g = M_inv;
  out.g_inv = M;
  return out;
}
