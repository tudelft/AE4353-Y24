# trajectory_simulation.py

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

# Define system dynamics
from additional.system_dynamics import f

def simulate(
    y0: float,
    z0: float,
    vy0: float,
    vz0: float,
    theta0: float,
    omega0: float,
    T: float,
    *,  # everything after this must be keyword-only
    controller
) -> dict:
    """
    Simulate the system dynamics over time.

    Parameters:
        y0, z0, vy0, vz0, theta0, omega0 : float
            Initial state variables.
        T : float
            Total simulation time (seconds).
        controller : callable
            Function that takes in the state vector (np.ndarray)
            and returns control output (np.ndarray).

    Returns:
        dict: Time series data for state variables and control inputs.
    """
    dt = 0.01
    t = np.linspace(0, T, int(T / dt))

    # Initialize states and control inputs
    x = np.array([y0, z0, vy0, vz0, theta0, omega0])
    X = [x]
    U = []

    # Simulation loop
    for _ in range(len(t) - 1):
        # Get control input
        u = controller(x)
        # Update state using dynamics function
        x = x + f(x, u) * dt
        # Store state and control input
        X.append(x)
        U.append(u)

    # Final control input
    u = controller(x)
    U.append(u)

    X = np.array(X)
    U = np.array(U)

    return {
        "t": t,
        "y": X[:, 0],
        "z": X[:, 1],
        "vy": X[:, 2],
        "vz": X[:, 3],
        "theta": X[:, 4],
        "omega": X[:, 5],
        "ul": U[:, 0],
        "ur": U[:, 1],
    }


def visualize_trajectory(
    y0=5, z0=5, vy0=10, vz0=5, theta0=2, omega0=2, T=5,
    app_name="test trajectory",
    *,  # everything after this must be keyword-only
    controller
):
    """
    Simulate and visualize a trajectory using rerun.

    Parameters:
        y0, z0, vy0, vz0, theta0, omega0, T : float
            Initial conditions and total simulation time.
        app_name : str
            Name for the rerun application.
        controller : callable
            Function that takes in a state vector (np.ndarray) and returns control output (np.ndarray).
    """
    traj = simulate(
        y0=y0, z0=z0, vy0=vy0, vz0=vz0,
        theta0=theta0, omega0=omega0, T=T,
        controller=controller,
    )

    rr.init(app_name)
    rr.log(
        "/world/Bounding Box",
        rr.Boxes2D(mins=[-10, -10], sizes=[20, 20]),
        timeless=True,
    )
    rr.log("/world/Destination", rr.Points2D([0, 0], colors=[0, 255, 0]), timeless=True)

    for i, t_val in enumerate(traj["t"]):
        rr.set_time_seconds("traj_time", t_val)
        rr.log("X/vy", rr.Scalar(traj["vy"][i]))
        rr.log("X/vz", rr.Scalar(-traj["vz"][i]))
        rr.log("X/theta", rr.Scalar(traj["theta"][i]))
        rr.log("X/omega", rr.Scalar(traj["omega"][i]))
        rr.log("U/ul", rr.Scalar(traj["ul"][i]))
        rr.log("U/ur", rr.Scalar(traj["ur"][i]))
        rr.log("/world/pos", rr.Points2D([traj["y"][i], -traj["z"][i]]))

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(
                origin="/world",
                time_ranges=[
                    rrb.VisibleTimeRange(
                        "traj_time",
                        start=rrb.TimeRangeBoundary.infinite(),
                        end=rrb.TimeRangeBoundary.cursor_relative(seq=0),
                    )
                ],
            ),
            rrb.Vertical(
                rrb.TimeSeriesView(origin="/X"),
            ),
        )
    )

    rr.send_blueprint(blueprint)
    rr.notebook_show()
