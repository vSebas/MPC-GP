#ifndef USV_DYNAMIC_MODEL_H
#define USV_DYNAMIC_MODEL_H

#include <Eigen/Dense>

#include <utility>

struct State {
  double x, y, psi;
  double u, v, r;
  double u_dot, v_dot, r_dot;
};

struct DecomposedDyn {
  Eigen::Vector3d f;
  Eigen::Matrix3d g, g_inv;
};

struct ModelParams {
  double m = 48.36; // [Kg]
  double xg = 0.0;  // Length from o_b to CG along x-axis
  double X_u_dot = -2.25, Y_v_dot = -23.13, Y_r_dot = -1.31, N_v_dot = -16.41,
         N_r_dot = -2.79; // Added masses params
  double Iz = 10.66;      // Moment of inertia

  // Nonlinear damping coefficients
  double Xuu = -70.92, Yvv = -99.99, Yrv = -5.49, Yvr = -5.49, Yrr = -8.8,
         Nvv = -5.49, Nrv = -8.8, Nvr = -8.8, Nrr = -3.49;

  double max_surge = 2.5;
  double max_astern = -1.5;
  double max_yaw = 3.0;

  // Update rate
  double integral_step = 0.01;
};

class UsvDynamicModel {
public:
  UsvDynamicModel(const Eigen::Vector3d &pose = Eigen::Vector3d::Zero(),
                  const Eigen::Vector3d &vel = Eigen::Vector3d::Zero(),
                  const ModelParams &params = ModelParams{});

  // Port / starboard thrusts [N]
  State update(double T_port, double T_stbd);
  // Generalized forces {X, Y, N}
  State update(Eigen::Vector3d u);
  State update_with_perturb(Eigen::Vector3d u, const Eigen::Vector3d &nu_c);
  // Surge force X and yaw moment N -> saturated {port, stbd} thrusts [N]
  static std::pair<double, double> allocate(double X, double N);
  static double wrap_angle(double angle);
  DecomposedDyn get_decomposed_dyn(const Eigen::Vector3d &nu_);

  constexpr static double B = 0.6; // Thruster separation [m]
  constexpr static double T_max = 36.5 * 2,
                          T_min = -30 * 2; // Per-thruster limits [N]

  static ModelParams usv_params() { return ModelParams{}; }

protected:
  ModelParams p;

  // State
  Eigen::Vector3d eta = Eigen::Vector3d::Zero();
  Eigen::Vector3d eta_dot_last = Eigen::Vector3d::Zero();
  Eigen::Vector3d nu = Eigen::Vector3d::Zero();
  Eigen::Vector3d nu_dot = Eigen::Vector3d::Zero();
  Eigen::Vector3d nu_dot_last = Eigen::Vector3d::Zero();
  Eigen::Matrix3d M, M_inv, C, D;
};

#endif // USV_DYNAMIC_MODEL_H
