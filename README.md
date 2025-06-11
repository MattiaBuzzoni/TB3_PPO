# End-to-end Motion Planner Using Proximal Policy Optimization (PPO) in PyBullet

The goal of this project is to replicate and adapt the approach presented in the paper ["Reinforcement Learning Based Mapless Navigation in Dynamic Environments Using LiDAR" (arXiv:2405.16266)](https://arxiv.org/abs/2405.16266) within the PyBullet simulation environment.

In this project, the authors introduce a learning-based mapless motion planner that utilizes sparse laser signals and the target's position in the robot frame (i.e., relative distance and angles) as inputs. This approach generates continuous steering commands as outputs, eliminating the need for traditional mapping methods like SLAM (Simultaneous Localization and Mapping). This planner is capable of navigating without prior maps and can adapt to new environments it has never encountered before.
This paper demonstrates the efficacy of deep reinforcement learning techniques in enabling robots to navigate complex, dynamic environments without relying on pre-built maps.
