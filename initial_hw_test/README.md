# Hardware and Communication Testing

This folder contains tests for different **hardware components** and **communication protocols** used in the project. It serves as the initial phase of development, experimenting with various approaches before integrating them into the full system.

## Overview

The purpose of this folder is to test various methods of communication and control for a robotic system. Several approaches were explored, including **serial communication**, **TCP sockets**, and **HTTP requests**. Additionally, different hardware setups were tested to assess their performance and reliability. The results from these experiments helped shape the later development of the project.

## File Descriptions

### Arduino Sketches (.ino)

#### `arduino2nmcu.ino`
This sketch establishes serial communication between an **Arduino** and a **NodeMCU (ESP8266)**, allowing the Arduino to send data to the NodeMCU. It was used to test the reliability of serial communication between the two devices.

#### `nmcu2arduino.ino`
Works as the opposite of `arduino2nmcu.ino`. The NodeMCU listens for serial input and sends back formatted responses. This helped test bidirectional communication.

#### `demo1.ino`
Controls a **line-following robot** equipped with **ultrasonic distance sensors**. The sketch uses the **NewPing** library to measure distances and adjusts movement accordingly to avoid obstacles. It was a foundational test for basic autonomous movement.

#### `test_arduino.ino`
Expands on `demo1.ino` by integrating **line tracking sensors** alongside motor control. The robot adjusts its direction based on line detection, allowing for more advanced navigation testing.

#### `ndemcu.ino`
This sketch sets up an **ESP8266 as a web server**, enabling LED control via HTTP requests. It serves as an initial test for remote control of the robot through a network interface.

#### `request.ino`
Instead of acting as a server, this sketch allows the **ESP8266** to send **HTTP GET requests** to a remote server. This was used to test data retrieval and command execution from a central web-based controller.

#### `test_nodemcu.ino`
Implements an **HTTP POST request system**, allowing the ESP8266 to send structured data to a remote server. It was tested as an alternative to the GET-based approach in `request.ino`.

#### `test_nodemcutcp.ino`
Unlike the previous web-based approaches, this sketch establishes a **TCP connection** for direct communication with a server. It tests sending and receiving commands in real-time with lower latency compared to HTTP.

#### `test_lt.ino`
This sketch focuses on **I2C communication**, requesting sensor data from a device at address `0x11`. The data is processed and converted into readable analog values, helping in hardware calibration and integration.

### Python Scripts (.py)

#### `demo1tcp.py`
Runs a **TCP server** on port `5000`, listening for connections from a client (such as an **Arduino or ESP8266**). The server processes **sensor data** (distance and line tracking) and determines movement commands to send back.

#### `demo1.py`
Provides an **HTTP-based control interface** using **Flask**. It receives movement requests from clients and processes sensor data to determine appropriate movement responses. This script was designed to run on a **Raspberry Pi or ESP8266** for web-based robot control.

#### `train.py`
Implements a **Deep Q-Learning (DQN) agent** using **TensorFlow/Keras**. The agent interacts with the robot via **TCP communication**, learning to follow a line and avoid obstacles through reinforcement learning. It serves as an initial training test for the robot. Various Q-learning strategies were implemented, including:
- **Standard Q-learning**
- **Fixed Q-target**
- **Double Q-network**

