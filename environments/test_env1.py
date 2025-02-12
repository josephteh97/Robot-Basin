import gymnasium as gym
imort matplotlib.pyplot as plt

# Create the MuJoCo environment
env = gym.make("Ant-v4", render_mode="rgb_array")   # Render as image array

obs, info = env.reset()

for _ in range(100):  # Run for 100 timesteps
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    # Render the environment to an image
    frame = env.render()

    # Display the frame using matplotlib
    plt.imshow()
    plt.axis("off")  # Hide axes
    plt.pause(0.05)  # Pause to simulate real-time display

    if terminated or truncated:
        obs, info = env.reset()

env.close()
