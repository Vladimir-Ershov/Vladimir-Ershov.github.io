---
layout: post
title: "BEHAVIOR Challenge @ NeurIPS 2025"
date: 2025-10-30
categories: robotics
author: Vladimir Ershov
description: "Initial review"
---

# What

benchmark with 1,000 defined household tasks grounded in real human needs
  50 of these full-length tasks in a realistic simulator - pushing the frontiers of both high-level planning and low-level control
  Task Definitions: BDDL
  50 high-fidelity, fully interactive scenes populated with around 10,000 objects
  OmniGibson Simulator: Built upon NVIDIA’s Omniverse - deformable objects (e.g. cloth and fabric), fluid interactions (pouring liquids), and complex object state changes (heating, cooling, cutting, etc.)
  Each task is a realistic scenario: (tidy bedroom”, “wash dog toys”, “collect children's toys”, “spray for bugs”, or “make a pizza”) several minutes of autonomous execution, multiple rooms of exploration
  10,000 teleoperated trajectories (over 1,200 hours of data)
  Galaxea’s R1 Pro robot, a wheeled humanoid
  no failed grasps, no accidental collisions with the environment, and no jittery, unnatural motions - only smooth and purposeful manipulation behavior
  
The goal is to complete the task as defined by the BDDL goal conditions (e.g., all the target objects in desired states and locations).

## Baselines

Classic Behavioral Cloning baselines: ACT, Diffusion Policy, BC-RNN, WB-VIMA
Pre-trained Visuo-Language Action models: OpenVLA and π0
12 GB limit


## Open Questions: 
- SimEnv detalisation (onion got autosliced)
- How many rooms? How extract the objects?
- System 2 / Teacher destiallation 
- IP based solution
