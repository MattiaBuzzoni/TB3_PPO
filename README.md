# End-to-end Motion Planner Using Proximal Policy Optimization (PPO) in PyBullet

The goal of this project is to replicate and adapt the approach presented in the paper ["Reinforcement Learning Based Mapless Navigation in Dynamic Environments Using LiDAR" (arXiv:2405.16266)](https://arxiv.org/abs/2405.16266) within the PyBullet simulation environment.

In this project, the authors introduce a learning-based mapless motion planner that utilizes sparse laser signals and the target's position in the robot frame (i.e., relative distance and angles) as inputs. This approach generates continuous steering commands as outputs, eliminating the need for traditional mapping methods like SLAM (Simultaneous Localization and Mapping). This planner is capable of navigating without prior maps and can adapt to new environments it has never encountered before.
This paper demonstrates the efficacy of deep reinforcement learning techniques in enabling robots to navigate complex, dynamic environments without relying on pre-built maps.

### Demo GIF

A demonstration of the mapless motion planner in action:

## Project Overview: Learning-Based Mapless Motion Planner
### Input Specifications (State):

The input features, forming a 16-dimensional state, include:

1. Laser Finding (10 Dimensions) - Represents sparse laser measurements.
2. Past Action (2 Dimensions):
  -Linear velocity
  -Angular velocity
3. Target Position in Robot Frame (2 Dimensions):
  -Relative distance
  -Relative angle (using polar coordinates)
4. Robot Yaw Angular (1 Dimension) - Indicates the robot's current yaw angle.
5. Degrees to Face the Target (1 Dimension) - The absolute difference between the yaw and the relative angle to the target.

### Normalization of Inputs:

Normalization is applied to the inputs to facilitate learning by scaling all values to a consistent range:

1. Laser Finding - Divided by the maximum laser finding range.
2. Past Action - Retained as original values.
3. Target Position in Robot Frame:
  -Relative distance normalized by the diagonal length of the map.
  -Relative angle normalized by 360 degrees.
4. Robot Yaw Angular - Normalized by 360 degrees.
5.  Degrees to Face the Target - Normalized by 180 degrees.


### Output Specifications (Action):

The outputs, forming a 2-dimensional action, consist of:

1. Linear Velocity (1 Dimension) - Ranging from 0 to 0.25 meters per second.
2. Angular Velocity (1 Dimension) - Ranging from -0.5 to 0.5 radians per second.

## Reward System
- Arrive at the target: +120
- Hit the wall: -100
- Move towards target: 500 * (Past relative distance - Current relative distance)

## Algorithm
- Proximal Policy Optimization (PPO) using Actor and Critic methods, implemented with PyTorch.

## Training Environment
PyBullet: A lightweight and versatile physics simulation environment for robotics and reinforcement learning, providing easy integration with Python-based machine learning frameworks such as PyTorch.

### Installation Dependencies:
1. Python3
2. PyTorch:
`pip3 install torch torchvision`
3. PyBullet
[http://wiki.ros.org/noetic/Installation/Ubuntu](https://pypi.org/project/pybullet/)

