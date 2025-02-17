# *Robot Cleaning Simulation (MuJoCo)*  

## *📌 Introduction*  
This project simulates a *robotic arm cleaning a face-washing bowl* using *MuJoCo* and *Reinforcement Learning (RL). The simulation provides a testing environment where the robot learns optimal cleaning motions using **machine learning algorithms*.  

---

## *📂 Project Structure*  

robot_cleaning_simulation/
│── models/                     # MuJoCo models (robot + bowl)

│   │── cleaning_robot.xml      # XML file defining the robot and environment

│── scripts/                    # Python scripts for control and learning
│   │── cleaning_env.py         # Custom Gym environment for the task
│   │── train_rl.py             # Reinforcement learning training script
│   │── eval.py                 # Model evaluation script

│── configs/                    # Configuration files
│   │── robot_config.yaml       # Robot movement settings
│   │── rl_config.yaml          # RL hyperparameters
│── logs/                       # Training performance logs

│── trained_models/             # Saved RL models
│── requirements.txt            # Dependencies
│── README.md                   # Project documentation


---

## *🔧 Installation*  
1. *Install dependencies*  
   bash
   pip install gym mujoco numpy stable-baselines3
   
2. *Ensure MuJoCo is properly installed*  
   bash
   pip install mujoco
   
3. *Clone this repository*  
   bash
   git clone https://github.com/yourusername/robot_cleaning_simulation.git
   cd robot_cleaning_simulation
   

---

## *🚀 Running the Simulation*  
### *1️⃣ Test the Environment*  
Run the following script to check if the simulation works properly:  
python
from cleaning_env import CleaningEnv

env = CleaningEnv()

obs = env.reset()

done = False

while not done:
    action = env.action_space.sample()  # Random action
    
    obs, reward, done, _ = env.step(action)
    
    env.render()

env.close()


### *2️⃣ Train an RL Agent (Optional)*  
If you want to train a reinforcement learning agent to optimize cleaning motions, run:  
python
python train_rl.py


---

## *📌 How It Works*  
- The *robot* is modeled in MuJoCo (cleaning_robot.xml).  
- The *environment* (cleaning_env.py) provides:
  - *Action Space* → Robot joint movements  
  - *Observation Space* → Joint states + cleanliness score + cleaned area vs expected area  
  - *Reward Function* → Increases as the bowl becomes cleaner    based on 1) collision 2) mis-cleaned area 3) dirt recognition
- The *robot learns* optimal cleaning motions using RL (e.g., PPO, SAC).  

---

## *📚 References*  
- [MuJoCo Documentation](https://mujoco.readthedocs.io/en/latest/)  
- [OpenAI Gym](https://gym.openai.com/)  
- [Stable-Baselines3 (RL Algorithms)](https://stable-baselines3.readthedocs.io/)  

---

## *🛠 Future Improvements*  
✅ Improve reward function for better cleaning efficiency  
✅ Add more realistic physics (e.g., water, soap interaction)  
✅ Implement a pre-trained motion planning strategy  

---

## *👨‍💻 Author*  
Zheng Jiezhi - Robotics Engineer  
Feel free to reach out or contribute to this project! 🚀  
use roboclean4 run
