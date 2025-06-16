import random
import rerun as rr
import rerun.blueprint as rrb
from torch.utils.data import Subset


def run_visualisation(val_set):
    # Initialize rerun
    rr.init("dataset")

    rr.log(
        "/world/Bounding_Box",
        rr.Boxes2D(mins=[-10, -10], sizes=[20, 20]),
        timeless=True,
    )
    rr.log(
        "/world/Destination",
        rr.Points2D([0, 0], colors=[0, 255, 0]),
        timeless=True,
    )

    # Get the first trajectory and control inputs from the validation set
    x_traj, u = val_set[0]

    # Log trajectory variables and controls for each time step
    for i, x in enumerate(x_traj):
        rr.set_time_sequence("traj_time", i)
        rr.log("X/vy", rr.Scalar(x[2]))  # Log vy, velocity in y
        rr.log("X/vz", rr.Scalar(x[3]))  # Log vz, velocity in z
        rr.log("X/theta", rr.Scalar(x[4]))  # Log theta, pitch angle
        rr.log("X/omega", rr.Scalar(x[5]))  # Log omega, angular velocity (pitch rate)
        rr.log("U/ul", rr.Scalar(u[i, 0]))  # Log ul, left motor command
        rr.log("U/ur", rr.Scalar(u[i, 1]))  # Log ur, right motor command
        rr.log("/world/pos", rr.Points2D([x[0], x[1]]))  # Log position in the world

    # Create a blueprint for visualization
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
            rrb.Vertical(rrb.TimeSeriesView(origin="/X"), rrb.TimeSeriesView(origin="/U")),
        )
    )

    # Show the visualization in the notebook
    rr.send_blueprint(blueprint)
    rr.notebook_show()