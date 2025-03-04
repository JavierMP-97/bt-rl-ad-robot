# Deep Reinforcement Learning for Autonomous Driving  

This repository is a time capsule of my **bachelor’s thesis** in 2019, where I explored whether **Deep Reinforcement Learning (DRL)** can be a viable tool for autonomous driving. Spoiler: it’s complicated. While it doesn’t quite work in its current state, this project is a **record of my evolution in software engineering and machine learning over a year**—complete with lessons learned, experiments logged, and a fair amount of debugging-induced existential crises.  

## Project Structure  

### `initial_hw_test` – Initial Hardware & Communication Testing  
Before jumping into the fancy AI stuff, I had to make sure the hardware didn’t catch fire. This step involved setting up microcontrollers, sensors, and communication protocols, testing interactions between Arduino, NodeMCU, TCP connections, HTTP requests, and I2C devices.  

### `line_follower_DQN` – Line Following with DQN  
My first attempt at Deep Q-Learning: an agent learns to follow a line using an infrared sensor. This was also my first DQN implementation, where I quickly learned that **experiment tracking is not optional**—trying to remember every configuration and tweak I made was like reconstructing a crime scene without security footage.  

### `simulator_DQN` – Advanced DQN in Simulation  
At this point, I dove deep into improving my DQN implementation by testing different ideas, sometimes breaking more things than I fixed. I experimented with variations like dueling Q-networks and double deep Q-networks, played around with preprocessing techniques like Canny edge detection, and fine-tuned the reward function to see how much better the agent could learn. Feature extraction became another key focus, and with each iteration, I gained a deeper appreciation for how small changes could have a big impact—or no impact at all, which was equally frustrating.  

### `simulator_SAC` – Soft Actor-Critic with an Improved Implementation  
This step was based on **[Learning to Drive Smoothly in Minutes](https://github.com/araffin/learning-to-drive-in-5-minutes/)**, which I modified using everything I had learned in the previous phase. Instead of relying on DQN, I transitioned to Soft Actor-Critic (SAC) to explore whether it could provide better stability and sample efficiency in a simulated driving environment.  

### `raspberry_SAC` – Deployment on a Real-World Model Car  
With all the knowledge gained from the previous steps, I attempted to train and deploy a reinforcement learning agent on a real, scaled-down self-driving car. Running on a Raspberry Pi with a camera as its only sensor, the goal was to translate everything I had learned in simulation into the real world—a task that, unsurprisingly, proved to be a lot harder than it sounded on paper.  

## Notes  
The experiment logs are included, so you can relive my journey of trial and error. However, the models and datasets are not included, as they are simply too large for the repository. While the project doesn’t quite work in its current form, it’s a testament to the learning process, the inevitable setbacks, and the small victories that made this journey worthwhile.



## 📄 Publication  
You can find the full thesis here: [Deep Reinforcement Learning for Autonomous Driving](https://hdl.handle.net/10016/30350)
