# DQN-Based Autonomous Driving in Simulation

This phase of the project explores **autonomous driving using Deep Q-Learning (DQN) with image-based input** in a simulated environment. Compared to the previous phase, where the agent relied on infrared sensors, this stage represents a significant **increase in complexity** as the agent must process raw **camera images** to make driving decisions. The simulator environment ensures that the approach is **tested and optimized** before transitioning to a real-world robotic vehicle.

This phase is based on the **Donkey Gym** implementation by **Tawn Kramer** ([GitHub Repo](https://github.com/tawnkramer/gym-donkeycar.git)) and the **reinforcement learning work of Felix Yu** ([Project](https://flyyufelix.github.io/2018/09/11/donkey-rl-simulation.html)).

## Overview

The goal of this phase is to develop a **self-learning AI driver** capable of:

- **Extracting relevant features from images** using preprocessing techniques.
- **Learning to steer and accelerate** based on visual input.
- **Generalizing across different track layouts** using procedural generation.
- **Testing optimal model architecture and hyperparameters** before real-world deployment.

A custom **reinforcement learning environment** was built on top of **Donkey Gym**, allowing seamless interaction between the **DQN agent** and the simulator.

## Watch the Project in Action

https://github.com/user-attachments/assets/d856c912-6357-4e55-9f7e-008724c6ee72

## Summary

### Problem Description

The agent must learn **autonomous driving** in a simulated environment using **image-based state representations**. Unlike previous phases where the agent relied on infrared sensors, this phase introduces:

- **A more complex state space**, where the model must interpret **visual input** instead of direct sensor values.
- **A continuous environment**, meaning the agent must learn smooth control rather than responding to discrete sensor signals.
- **Dynamic road conditions**, requiring generalization across multiple track layouts.

The main challenge is enabling the agent to **perceive lane markings** and **make driving decisions** without predefined rules.

The objective of this task is to train an autonomous driving agent that can navigate a simulated track for 2000 consecutive steps without exiting the road.&#x20;

### Model Design

#### **Image Preprocessing**

To extract meaningful information, the camera input undergoes several preprocessing steps:

1. **Captured images:** 160×120 pixels.
2. **Cropped** to remove unnecessary background, leaving **160×80 pixels**.
3. **Converted to grayscale** to reduce computational complexity.
4. **Gaussian noise and Canny edge detection** applied to highlight road boundaries.
5. **Stack of the last 3 frames** used as input to provide a sense of motion.

![imagen DQN](https://github.com/user-attachments/assets/c4245265-976f-455d-8ddb-3dbb7606f76c)

#### **State Representation**

Each state consists of a **3-frame stack** of processed images. The model optionally includes **speed data** to improve learning stability.

#### **Action Space**

- **Phase 1:** Steering-only control (7 discrete actions).
- **Phase 2:** Steering + acceleration (15 discrete actions, combining 3 acceleration levels and 5 steering angles).

#### **DQN Architecture**

- **Input shape:** `[80, 160, 1, 3]` (height, width, channels, stacked frames).
- **Convolutional layers:**
  - **3D convolution** with **8 filters**, kernel size **3×3**, stride **2**.
  - Followed by **MaxPooling**.
- **Fully connected layer:** 50 neurons.
- **Output layer:** Neurons equal to the number of actions.
- **Activation functions:**
  - **ReLU** for hidden layers, **linear** for output.
- **Weight initialization:** `he_normal`.
- **Optimizer:** Adam with **gradient clipping (1.0)**.

The model is optimized for real-time execution, achieving **25 FPS on a Raspberry Pi 3B**.

### Reward Model

The reward function is based on the **Cross-Track Error (CTE)**, which measures the car's distance from the center of the lane:

- **Positive reward:**\
  \((1 – CTE² / CTE_{max}) \times \frac{\text{speed}}{30}\)\
  (Encourages staying close to the center and maintaining speed).
- **Penalty of -1:**
  - If CTE exceeds `CTE_max` (1.5), meaning the car has gone off track.
  - If the car remains stationary for more than **10 ticks**.
- **Episode ends if:**
  - The car exceeds **CTE\_max**.
  - The car completes a full lap without leaving the track.

### Training Implementation

#### **1. Experience Replay**

The agent stores past experiences (`(state, action, reward, next_state)`) in a **memory buffer** and randomly samples mini-batches to train the model. This prevents overfitting to recent experiences and stabilizes training.

#### **2. Double Target Q-Network**

A **secondary Q-network** is used to compute target values, reducing Q-value overestimation and improving learning stability.

#### **3. Procedural Track Generation**

- At the start of training, the **track seed is set to 0**.
- After each episode, the seed is **incremented**, generating a **new randomized track**.
- This ensures **consistent training while promoting generalization**.

#### **4. Testing Between Training Episodes**

- Every training episode is followed by a **test episode**, where the agent’s performance is evaluated **without policy updates**.
- This provides a **reliable metric of progress** and prevents misleading results caused by short-term learning fluctuations.

### Experimental Results

#### **Phase 1: Steering Prediction Only**

- The agent controls **only the steering**, with a constant speed.
- Using the **Canny edge detector** greatly improved training efficiency.
- **Training stability issues** were common due to DQN’s sensitivity to hyperparameters.

#### **Phase 2: Steering + Acceleration Control**

- Introducing acceleration control made the problem significantly more complex.
- **Double Target Q-Network** improved performance but did not eliminate training instability.
- **Speed control was inconsistent**, with the agent often approaching corners too fast.

![reward_otros](https://github.com/user-attachments/assets/52adb265-4d71-4305-b7f5-366791ef4f49)

### Conclusion

This phase successfully demonstrated that an RL-based agent can learn autonomous driving using images. However, several challenges emerged, highlighting the limitations of Deep Q-Learning for continuous control tasks:

- **Training instability remained a major issue.** The agent's performance fluctuated unpredictably, sometimes regressing despite prior progress.
- **Hyperparameter tuning was critical.** A lower learning rate was necessary compared to previous phases.
- **Feature extraction made a huge difference.** The Canny edge detector significantly accelerated learning.
- **Full-range acceleration control was unnecessary.** Limiting braking improved training efficiency.
- **More discrete actions didn’t help.** The agent still relied on a small subset, wasting available options.

Given these findings, a more robust solution is needed—one that allows continuous action spaces instead of discretized steering/acceleration and more stable training algorithms that produce consistent results.

This phase provided valuable insights into the limitations of DQN for autonomous driving, paving the way for more advanced reinforcement learning approaches in future phases.

Beyond the technical challenges, this phase drove home some important lessons. Experiment tracking is not optional—this time, I actually logged experiments and tracked key metrics, and it turns out that writing things down works (who would have thought?). Relying on memory for hyperparameters is a fantastic way to repeat past mistakes without realizing it. Being methodical wasn’t just about organization; structured tests made it possible to isolate the real impact of each change instead of guessing what went wrong. And most importantly, never trust the training policy blindly. Just because a policy works during training doesn’t mean it will hold up when evaluated deterministically. Running tests on the deterministic policy after training was essential to understanding what the model had actually learned.


