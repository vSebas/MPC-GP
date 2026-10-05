from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos, fabs, atan2
import casadi as ca
import numpy as np


class USVAcadosModel(AcadosModel):
    T_max: float
    T_min: float


def export_usv_model() -> USVAcadosModel:
    model_name = "usv_dynamics"

    m = 48.36
    B = 0.6
    Iz = 10.66
    xg = 0.0

    # Added mass
    X_u_dot = -2.25
    Y_v_dot = -23.13
    Y_r_dot = -1.31
    N_v_dot = -16.41
    N_r_dot = -2.79

    # Nonlinear damping
    Xuu = -70.92
    Yvv = -99.99
    Yvr = -5.49
    Yrv = -5.49
    Yrr = -8.8
    Nvv = -5.49
    Nvr = -8.8
    Nrv = -8.8
    Nrr = -3.49

    # Thruster limits for normalization
    T_max = 36.5 * 2
    T_min = -30.0 * 2

    # Inertia matrix M (constant) — precomputed numerically
    M_np = np.array(
        [
            [m - X_u_dot, 0.0, 0.0],
            [0.0, m - Y_v_dot, m * xg - Y_r_dot],
            [0.0, m * xg - N_v_dot, Iz - N_r_dot],
        ]
    )
    M_inv_np = np.linalg.inv(M_np)

    # =========================================================================
    # States: [x, y, psi, surge, sway, yaw]
    # =========================================================================
    x_pos = SX.sym("x_pos")
    y_pos = SX.sym("y_pos")
    psi = SX.sym("psi")
    surge = SX.sym("surge")
    sway = SX.sym("sway")
    yaw = SX.sym("yaw")

    # Thrust commands
    u_Tport = SX.sym("u_Tport")
    u_Tstbd = SX.sym("u_Tstbd")

    x = vertcat(x_pos, y_pos, psi, surge, sway, yaw, u_Tport, u_Tstbd)

    # =========================================================================
    # Controls: port/starboard thrust rates
    # =========================================================================
    du_Tport = SX.sym("du_Tport")
    du_Tstbd = SX.sym("du_Tstbd")

    u_ctrl = vertcat(du_Tport, du_Tstbd)

    # =========================================================================
    # Parameters
    # =========================================================================
    p_dummy = SX.sym("p_dummy")

    p = vertcat(
        p_dummy,
    )

    # =========================================================================
    # Coriolis matrix C(nu) — matching C++ get_decomposed_dyn()
    # =========================================================================
    c0 = m * (xg * yaw + sway)
    c1 = m * surge
    C_RB = SX.zeros(3, 3)
    C_RB[0, 2] = -c0
    C_RB[1, 2] = c1
    C_RB[2, 0] = c0
    C_RB[2, 1] = -c1

    c2 = Y_v_dot * sway + 0.5 * (Y_r_dot + N_v_dot) * yaw
    c3 = X_u_dot * surge
    C_A = SX.zeros(3, 3)
    C_A[0, 2] = c2
    C_A[1, 2] = -c3
    C_A[2, 0] = -c2
    C_A[2, 1] = c3

    C_mat = C_RB + C_A

    # =========================================================================
    # Damping matrix D(nu)
    # =========================================================================
    surge_abs = fabs(surge)
    sway_abs = fabs(sway)
    yaw_abs = fabs(yaw)

    vel = ca.sqrt(surge**2 + sway**2 + 1e-9)

    # Linear damping Dl
    Xu = ca.if_else(surge > 1.2, 64.55, -25.0)
    Xuu_act = ca.if_else(surge > 1.2, Xuu, 0.0)

    k = 0.09 * 0.09 * 1.01
    Yv = (
        0.5
        * (-40000.0 * sway_abs)
        * (
            1.1
            + 0.0045 * (1.01 / 0.09)
            - 0.1 * (0.27 / 0.09)
            + 0.016 * (0.27 / 0.09) ** 2
        )
    )
    Yr = 6.0 * (-np.pi * 1000.0) * vel * k
    Nv = 0.06 * (-np.pi * 1000.0) * vel * k
    Nr = 0.02 * (-np.pi * 1000.0) * vel * k * 1.01

    # --- D = Dl - Dn ---
    D_mat = SX.zeros(3, 3)
    D_mat[0, 0] = -Xu - Xuu_act * surge_abs
    D_mat[1, 1] = -Yv - (Yvv * sway_abs + Yvr * yaw_abs)
    D_mat[1, 2] = -Yr - (Yrv * sway_abs + Yrr * yaw_abs)
    D_mat[2, 1] = -Nv - (Nvv * sway_abs + Nvr * yaw_abs)
    D_mat[2, 2] = -Nr - (Nrv * sway_abs + Nrr * yaw_abs)

    # =========================================================================
    # Thrust, physical forces
    # =========================================================================
    T_SCALE = T_max
    Tp, Ts = u_Tport * T_SCALE, u_Tstbd * T_SCALE
    tau_thrust = vertcat(Tp + Ts, 0, 0.5 * B * (Tp - Ts))

    # =========================================================================
    # nu_dot = M^{-1} * (tau - C*nu - D*nu)
    # =========================================================================
    nu_vec = vertcat(surge, sway, yaw)
    M_inv_sx = SX(M_inv_np)
    nu_dot = M_inv_sx @ (tau_thrust - C_mat @ nu_vec - D_mat @ nu_vec)

    # =========================================================================
    # Kinematics
    # =========================================================================
    cos_psi = cos(psi)
    sin_psi = sin(psi)

    # =========================================================================
    # Full explicit ODE
    # =========================================================================
    f_expl = vertcat(
        surge * cos_psi - sway * sin_psi,
        surge * sin_psi + sway * cos_psi,
        yaw,
        nu_dot[0],
        nu_dot[1],
        nu_dot[2],
        du_Tport,
        du_Tstbd,
    )

    # Implicit form
    x_dot_sym = SX.sym("x_dot")
    y_dot_sym = SX.sym("y_dot")
    psi_dot_sym = SX.sym("psi_dot")
    surge_dot = SX.sym("surge_dot")
    sway_dot = SX.sym("sway_dot")
    yaw_dot = SX.sym("yaw_dot")
    xdot = vertcat(
        x_dot_sym,
        y_dot_sym,
        psi_dot_sym,
        surge_dot,
        sway_dot,
        yaw_dot,
        SX.sym("u_Tport_dot"),
        SX.sym("u_Tstbd_dot"),
    )

    f_impl = xdot - f_expl

    # =========================================================================
    # Assemble model
    # =========================================================================
    model = USVAcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u_ctrl
    model.p = p
    model.name = model_name

    model.x_labels = [
        "$x$ [m]",
        "$y$ [m]",
        "$\\psi$ [rad]",
        "$u$ [m/s]",
        "$v$ [m/s]",
        "$r$ [rad/s]",
    ]
    model.u_labels = [
        "$\\dot{u}_{Tport}$",
        "$\\dot{u}_{Tstbd}$",
    ]

    model.T_max = T_max
    model.T_min = T_min

    return model
