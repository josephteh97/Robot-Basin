import mujoco
import numpy as np
import matplotlib.pyplot as plt

# Load the MuJoCo model
model = mujoco.MjModel.from_xml_path('environments/ant.xml')  # Update with the path to your MuJoCo model XML file
# model = mujoco.MjModel.from_xml_path('models/ec63_description.urdf')
data = mujoco.MjData(model)

# Create a MuJoCo viewer
viewer = mujoco.MjViewer(data)

# Simulation parameters
num_steps = 100

for _ in range(num_steps):
    # Random action
    action = np.random.uniform(model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1], model.nu)
    data.ctrl[:] = action

    # Step the simulation
    mujoco.mj_step(model, data)

    # Render the current frame
    viewer.render()

    # Capture the frame as an image
    rgb_array = viewer.read_pixels(width=viewer.viewport.width, height=viewer.viewport.height)

    # Display the frame using matplotlib
    plt.imshow(rgb_array)
    plt.axis("off")  # Hide axes
    plt.pause(0.05)  # Pause to simulate real-time display

mujoco.mj_close()