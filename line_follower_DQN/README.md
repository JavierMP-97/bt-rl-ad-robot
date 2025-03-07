# DQN Line-Following Robot

This folder contains the implementation of a **Deep Q-Network (DQN) agent** that enables a robot to follow a line using **infrared sensors**. The system integrates **reinforcement learning**, **sensor calibration**, and **TCP communication** to process real-time sensor data and determine movement actions.

## Overview

This section of the project contains a **learning-based control system** for a line-following robot. The approach involves:
- **Infrared sensor calibration** to normalize sensor readings.
- **A deterministic rule-based agent** for initial training.
- **A Deep Q-Network (DQN) agent** that learns optimal movement strategies.
- **TCP communication** to send and receive commands between the robot and the server.
- **Arduino and ESP8266 firmware** for real-time sensor reading and motor control.

## Watch the Project in Action
https://github.com/user-attachments/assets/3d00d339-dd72-45a9-b68f-b45d38ef30a7

## File Descriptions

### Python Scripts

#### `calibrate.py`
This script performs **infrared sensor calibration**. The robot sends **raw sensor readings** over a **TCP connection**, and the script calculates the range of values for both the **floor and the line**. The calibration data is saved to a file (`calibration.pr`) for later use in normalization.

#### `demo1tcp.py`
A **TCP server** that processes **real-time sensor data** from the robot. It normalizes the readings using calibration data, determines the **robot’s position relative to the line**, and decides an appropriate action. The action is sent back to the robot for execution.

#### `DeterministicAgent.py`
Implements a **rule-based agent** that makes movement decisions based on sensor readings. Unlike the DQN agent, it does not learn from experience but follows **pre-defined rules** to keep the robot on track. This agent is used for generating **pre-training data** before reinforcement learning is applied.

#### `DQN.py`
The main script that runs the **Deep Q-Network (DQN) training loop**. It:
- Initializes a **DQNAgent**
- Loads pre-trained data if available
- Starts a **TCP server** to interact with the robot
- Uses **reinforcement learning** to improve decision-making

The training includes **experience replay**, an **epsilon-greedy policy** for action selection, and **state normalization** based on calibration data.

#### `DQNAgent.py`
Defines the **DQN agent**, which learns how to follow a line using **infrared sensors**. It utilizes:
- **A deep neural network** to approximate the Q-function.
- **Reinforcement learning strategies**, including:
  - **Standard Q-learning**
  - **Fixed Q-target**
  - **Double Q-network**
- **Huber loss** for stable training.
- **Experience replay** to improve learning efficiency.

#### `pre_train.py`
This script pre-trains the DQN agent using a **DeterministicAgent**. It generates **training data** from rule-based decisions, allowing the DQN agent to start learning with **pre-recorded state-action pairs** before interacting with the real environment. It communicates with the robot over **TCP** to collect sensor readings and actions.

### Arduino Sketches

#### `nodemcu_tcp.ino`
Runs on an **ESP8266 NodeMCU** and acts as a **TCP client**. It:
- Reads **infrared sensor data** from the robot.
- Sends the data to the **Python-based TCP server**.
- Receives an action decision and forwards it to the **motor controller**.

#### `arduino_train.ino`
Deployed on an **Arduino** connected to the robot’s **motors and infrared sensors**. It:
- Reads sensor values and transmits them to the **ESP8266**.
- Executes movement commands received from the **DQN agent** via **TCP**.

### Usage

1. **Calibrate the sensors**
   - Run `calibrate.py` while the robot is placed on the floor and the line.
   - Save the calibration data.

2. **Run the Deterministic Agent (Optional Pre-training)**
   - Execute `pre_train.py` to collect training data using the rule-based approach.

3. **Train the DQN agent**
   - Start `DQN.py` to train the agent in real-time.
   - The robot will move and learn based on sensor inputs.

4. **Deploy the trained model**
   - Once trained, the agent can be used directly with `DQN.py` for inference.
  
## Summary

### Problem Description

The task involves developing a **self-learning robotic vehicle** that can autonomously follow a black line using **reinforcement learning (RL)**. The agent must learn to:
- Detect the line using **infrared sensors**.
- Adjust motor speeds to stay on the track.
- Recover if it deviates from the path.
- Optimize its movement efficiency.
- Complete a full lap around the circuit


![IMG_20190919_045909](https://github.com/user-attachments/assets/ab1073b0-aca2-41d4-9b23-1d51c9752f04)


### State Representation
- 5 continious values to represent the state of each sensor
- Last detected sensor (values that goes from -1 to 1, depending on which was the last IR sensor detected)

The last detected sensor helps the agent **recover** when it loses track of the line.

### Action Space
- 7 actions that allow for granular turning options

### Reward Model
- **+15** if the center sensor detects the line.
- **+10** if an inner sensor detects it.
- **+5** if an outer sensor detects it.
- **0** if no sensors detect the line.
- **Episode ends if the line is not detected for 20 ticks**.

### Training Implementation Key Details

#### 1. Experience Replay
The agent stores previous experiences (`(state, action, reward, next_state)`) in a **memory buffer** and randomly samples mini-batches to train the model. This prevents overfitting to recent experiences and stabilizes learning.

#### 2. Double Target Q-Network
To reduce overestimation of Q-values, **a secondary Q-network** is used to compute target values, improving training stability.

#### 3. Auto-Recovery Mechanism
If the robot **loses track of the line** (ending the episode), it enters a **recovery mode**, where it **moves backward and searches for the track** using its last detected sensor position. This prevents the agent from getting stuck and improves learning efficiency.

#### 4. Pre-Training with a Deterministic Agent
Before training the **DQN agent**, a **rule-based agent** is used to generate a dataset of **state-action pairs**. This allows the DQN model to start learning from a **pre-trained policy**, improving convergence speed.

#### 5. TCP-Based Low-Latency Communication
Initially tested with **HTTP**, but later switched to **TCP sockets** for faster message exchange. This ensures minimal delay between **sensor readings and action execution**.

## Experimental Results
- **Experience Replay** was essential; without it, the agent failed to complete the track.
- **Double Target Q-Network** improved stability but did not drastically outperform the default setup.
- **Using a deeper network** improved learning, but **the DQN was highly sensitive to hyperparameters**.

## Conclusion

This stage wrapped up successfully, laying the groundwork for the entire project. It validated the hardware and communication setup, ensured the training implementation worked, and proved that the robot could learn to follow a line instead of wandering off like a confused Roomba.

One crucial lesson learned: track every experiment meticulously. While trial and error is great for reinforcement learning, it turns out humans don't have experience replay built-in. Remembering every configuration and hyperparameter tweak is impossible, so proper logging became a must. Otherwise, every experiment starts feeling like Déjà Vu.

