import numpy as np

def simulate(
    y0: float, z0: float, vy0: float, vz0: float, theta0: float, omega0: float, T: float
) -> dict:
    """
    Simulate the system dynamics over time.

    Parameters:
        y0 (float): Initial y position.
        z0 (float): Initial z position.
        vy0 (float): Initial y velocity.
        vz0 (float): Initial z velocity.
        theta0 (float): Initial angle.
        omega0 (float): Initial angular velocity.
        T (float): Total simulation time.

    Returns:
        dict: A dictionary containing time series data for state variables and control inputs.
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

    # Compute final control input
    u = controller(x)
    U.append(u)
    X = np.array(X)
    U = np.array(U)

    # Return simulation results
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