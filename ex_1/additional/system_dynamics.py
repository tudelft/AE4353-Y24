import numpy as np


def f(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Compute the dynamics of the system given the state and control input.

    Parameters:
        x (np.ndarray): The current state vector [y, z, vy, vz, theta, omega].
        u (np.ndarray): The control input vector [ul, ur].

    Returns:
        np.ndarray: The derivatives of the state vector as a numpy array.
    """
    # Unpack state variables
    y, z, vy, vz, theta, omega = x

    # Unpack control variables
    ul, ur = u

    # Define parameters
    g = 9.81  # Gravity (m/s^2)
    m = 0.389  # Mass (kg)
    Ixx = 0.001242  # Moment of inertia (kg*m^2)
    L = 0.08  # Length (m)
    maxthrust = 2.35  # Maximum thrust (N)
    minthrust = 1.76  # Minimum thrust (N)
    M = maxthrust - minthrust  # Thrust range (N)
    F = 2 * minthrust  # Base thrust (N)
    beta = 0.5  # Drag coefficient (-)

    # Compute dynamics
    dy_dt = vy
    dz_dt = vz
    dv_ydt = -((ur + ul) / m * M + F / m) * np.sin(theta) - beta * vy
    dvz_dt = ((ur + ul) / m * M + F / m) * np.cos(theta) - g - beta * vz
    dtheta_dt = omega
    domega_dt = L / Ixx * M * (ur - ul)

    return np.array([dy_dt, dz_dt, dv_ydt, dvz_dt, dtheta_dt, domega_dt])