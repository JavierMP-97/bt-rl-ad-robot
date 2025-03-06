# SAC-Based Autonomous Driving in Simulation

This phase of the project explores **autonomous driving using Soft Actor-Critic (SAC)**, a reinforcement learning algorithm optimized for continuous action spaces, combined with a **Variational Autoencoder (VAE)** for stable image encoding in a simulated environment. Compared to the previous DQN-based phase, this stage represents a significant **increase in performance and training stability**, allowing the agent to learn smoother, more realistic driving behaviors. The simulation environment ensures the approach is thoroughly **tested and optimized** before transitioning to real-world robotic vehicles.

This phase builds upon **Antonin Raffin's** implementation of SAC in the **Donkey Gym** simulator ([GitHub Repo](https://github.com/araffin/learning-to-drive-in-5-minutes)), leveraging **Stable Baselines** and **Donkey Gym** with modifications for integrating a VAE.

## Overview

The goal of this phase is to develop a **self-learning AI driver** capable of:

- **Extracting stable, compact latent features** from raw camera images using a pretrained VAE.
- **Learning continuous steering and acceleration control**, resulting in smoother vehicle motion.
- **Generalizing effectively** across procedurally generated track layouts.
- **Testing optimal model architecture and hyperparameters** rigorously before real-world deployment.

A custom **reinforcement learning environment** was built on top of **Donkey Gym**, ensuring seamless interaction between the SAC agent, the pretrained VAE encoder, and the simulator.

## Watch the Project in Action

https://github.com/user-attachments/assets/ca18fde2-6d31-41e7-ab39-3f92ed40beca

## Summary

### Problem Description

The agent must learn **autonomous driving** in a simulated environment using **image-based state representations**. Unlike previous phases with discrete DQN-based actions, this phase introduces:

- **Continuous action spaces** enabling smoother, realistic vehicle control.
- **Latent image encoding** via a pretrained VAE to reduce state complexity and stabilize learning.
- **Dynamic road conditions**, requiring robust generalization across multiple randomized tracks.

The primary challenge is enabling the agent to **interpret visual cues from lane markings** and drive continuously without predefined rules.

The objective is to train an autonomous agent capable of navigating procedural tracks for **3000 consecutive steps** without exiting the road.

### Model Design

#### Image Preprocessing and VAE Encoding

Camera images undergo the following preprocessing:

1. **Captured images:** 160×120 pixels.
2. **Cropped** to remove unnecessary background, yielding **160×80 pixels**.
3. **Encoded** into a compact **32-dimensional latent vector** via a pretrained Variational Autoencoder.

![VAE sim](https://github.com/user-attachments/assets/db79bbb9-9cae-4dfe-b224-d2c753112019)

The VAE architecture includes:

- **Input shape:** `[80, 160, 3]` RGB images.
- **Convolutional layers:** 4 layers with [32, 64, 128, 256] filters, kernel size `4×4`, stride `2`.
- **Output:** Latent vector of size **32**.
- **Activation function:** ReLU.

This VAE encoding achieves real-time inference, performing at **over 10 FPS on a Raspberry Pi 3B**.

#### State Representation

The state consists of:

- **32-dimensional latent vector** from the VAE.
- Last **20 actions performed** (steering and acceleration), totaling **40 values**.

The final input to the SAC networks is a **72-dimensional vector**.

#### Action Space

Continuous action space controlling two dimensions:

- **Steering angle**
- **Acceleration**

Additionally, steering commands are limited to incremental adjustments of **±15%**, ensuring smoother vehicle trajectories and reducing zigzagging behavior.

#### SAC Architecture

**Actor and Critic Networks:**

- **Input:** 72 neurons (latent vector + action history), plus action inputs for the critic network.
- **Hidden layers:** Two layers with 32 and 16 neurons.
- **Output:** 
  - **Actor:** 2 neurons (steering, acceleration).
  - **Critic:** 1 neuron (estimated action-state value).
- **Activation functions:**
  - Hidden layers: **ELU**
  - Output layers: **Linear**

The compact architecture allows efficient real-time inference suitable for deployment on limited hardware.

### Reward Model

The reward function encourages smooth and centered driving behavior:

- **Positive reward (+1)** per timestep on track.
- **Additional reward** proportional to normalized acceleration (**weight = 0.1**).
- **Penalty (-10)** plus additional acceleration penalty if the car exceeds **maximum Cross-Track Error (CTE)**.
- Episode terminates if:
  - **CTE exceeds maximum threshold**
  - Agent completes a lap or reaches **3000 ticks**

### Training Implementation

#### 1. Experience Replay

Experiences (`state, action, reward, next_state`) are stored in a memory buffer, randomly sampling mini-batches for training, which stabilizes the learning process.

#### 2. Soft Actor-Critic Algorithm (Stable Baselines)

The **Stable Baselines** library provides the SAC implementation, facilitating stable and robust continuous-action reinforcement learning.

#### 3. Procedural Track Generation

- **Initial seed:** 0
- Seed increments after each episode, generating randomized tracks to ensure effective generalization.

### Experimental Results

#### SAC Performance Highlights

- **Rapid training convergence:** Agent consistently reaches stable driving within **30–120 episodes**.
- **Robust to hyperparameter variations:** Performance remained stable despite considerable hyperparameter adjustments.
- **VAE-based encoding** proved critical to training stability. Direct convolutional encoding without a VAE resulted in instability comparable to DQN.
- Allowing reverse acceleration led to overly cautious policies, emphasizing careful action-space definition.
- Lowering the **CTE_max** improved lane-centering without reducing overall performance.

### Conclusion

This phase demonstrated clearly that the **Soft Actor-Critic (SAC)** algorithm is superior to DQN for continuous-action autonomous driving tasks:

- **Stable training:** Significant improvement in predictability and stability compared to DQN.
- **Smooth control:** Continuous actions enabled realistic vehicle movements with minimal zigzagging.
- **Essential VAE encoding:** Stabilized the input representation, crucial for consistent learning outcomes.
- **Robustness to hyperparameters** significantly reduced tuning complexity compared to the previous phase.

Beyond technical improvements, this phase showed signs that my research methodology was finally becoming passable (it only took a few phases...). Although my direct contributions at this stage were smaller, I gained a much deeper understanding of complex, math-heavy algorithms such as SAC and VAEs. SAC and VAEs turned out to be far less intuitive than good old DQN and classic convolutional networks; apparently, entropy-regularized value functions and latent-space encodings don't exactly lend themselves to "eyeballing" solutions. Nevertheless, this phase taught me how to effectively approach and learn these challenging concepts, setting me up nicely for tackling real-world robot training in the final stage.

Overall, this phase laid a solid foundation, demonstrating the effectiveness of SAC combined with latent representations for stable and robust reinforcement learning. These results provide a clear path forward, preparing the groundwork for successfully transitioning from simulation-based autonomous driving to real-world robot deployment in the next stage.
