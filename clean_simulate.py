import numpy as np
import matplotlib.pyplot as plt
from cleaning_env import CleaningEnv

# Parameters
num_episodes = 10
max_steps_per_episode = 100

# Initialize environment
env = CleaningEnv(xml_path="cleaning_robot.xml")

# Run simulation and collect data
cleanliness_scores = []

for episode in range(num_episodes):
    observation = env.reset()
    episode_scores = []

    for step in range(max_steps_per_episode):
        action = np.random.uniform(low=-1.0, high=1.0, size=env.action_space['shape'])  # Random actions
        observation, reward, done, info = env.step(action)
        episode_scores.append(observation[-1])  # Cleanliness score is the last observation element

        if done:
            break

    cleanliness_scores.append(episode_scores)

# Close environment
env.close()

# Plot results
for i, scores in enumerate(cleanliness_scores):
    plt.plot(scores, label=f'Episode {i+1}')

plt.xlabel('Steps')
plt.ylabel('Cleanliness Score')
plt.title('Cleanliness Score Over Time')
plt.legend()
plt.show()