# SPDX-FileCopyrightText: Copyright (c) 2022 Guillaume Bellegarda. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2022 EPFL, Guillaume Bellegarda

import os, sys
import gym
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from sys import platform
# may be helpful depending on your system
# if platform =="darwin": # mac
#   import PyQt5
#   matplotlib.use("Qt5Agg")
# else: # linux
#   matplotlib.use('TkAgg')

# stable-baselines3
from stable_baselines3.common.monitor import load_results 
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import PPO, SAC
# from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.env_util import make_vec_env # fix for newer versions of stable-baselines3

from env.quadruped_gym_env import QuadrupedGymEnv
# utils
from utils.utils import plot_results
from utils.file_utils import get_latest_model, load_all_results


LEARNING_ALG = "SAC"
interm_dir = "./logs/intermediate_models/"
# path to saved models, i.e. interm_dir + '121321105810'
log_dir = interm_dir + '010624162809' #121723153820 <LR_COURSE> 121823121519 <FLAGRUN> 121723233147 <CPG>

# initialize env configs (render at test time)
# check ideal conditions, as well as robustness to UNSEEN noise during training
env_config = {}
env_config['render'] = True
env_config['record_video'] = False
env_config['add_noise'] = True 
env_config['motor_control_mode'] = "CARTESIAN_PD"
env_config['observation_space_mode'] = "LR_COURSE_OBS"
env_config['task_env'] = "LR_COURSE_TASK"
env_config['competition_env'] = False
# env_config['test_env'] = True

# get latest model and normalization stats, and plot 
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
model_name = get_latest_model(log_dir)
monitor_results = load_results(log_dir)
print(monitor_results)
plot_results([log_dir] , 10e10, 'timesteps', LEARNING_ALG + ' ')
plt.show() 

# reconstruct env 
env = lambda: QuadrupedGymEnv(**env_config)
env = make_vec_env(env, n_envs=1)
env = VecNormalize.load(stats_path, env)
env.training = False    # do not update stats at test time
env.norm_reward = False # reward normalization is not needed at test time

# load model
if LEARNING_ALG == "PPO":
    model = PPO.load(model_name, env)
elif LEARNING_ALG == "SAC":
    model = SAC.load(model_name, env)
print("\nLoaded model", model_name, "\n")

obs = env.reset()
episode_reward = 0
time_passed = 0
time_up = 0
time_down = 0
TIME_STEP = 0.001
time_down_array = []
time_up_array = []
foot_positions = {i: {'x': [], 'z': []} for i in range(4)} 
r_values = []
rdot_values = []
theta_values = []
theta_dot_values = []
# [TODO] initialize arrays to save data from simulation 
base_velocity = []
energy = 0
for i in range(5000):
    time_passed = env.envs[0].env.get_sim_time()
    action, _states = model.predict(obs,deterministic=False) # sample at test time? ([TODO]: test)
    obs, rewards, dones, info = env.step(action)
    episode_reward += rewards
    # print("base pos", info[0]['base_pos'][2])
    if dones:
        print('episode_reward', episode_reward)
        print('Final base position', info[0]['base_pos'])
        print('Time passed', time_passed)
        COT = energy / (info[0]['base_pos'][0] * 9.81 * 12)
        print("Cost of Transport: ",COT)
        print("Average Velocity: ", np.mean(base_velocity))
        print("Average Stance Time: ", np.mean(time_down_array))
        print("Average Swing Time: ", np.mean(time_up_array))
        episode_reward = 0
        energy = 0
        plt.figure()
        for i in range(4):  # Assuming 4 legs
            plt.plot(foot_positions[i]['x'], foot_positions[i]['z'], label=f'Foot {i+1}', linewidth=0.2)
        plt.xlabel('x (m)')
        plt.ylabel('z (m)')
        plt.axis("equal")
        plt.figure()
        plt.plot(base_velocity, label="values")
        plt.plot([np.mean(base_velocity)]*len(base_velocity), label="mean", linestyle="--")
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s)')
        plt.title('Velocity')
        if env_config['motor_control_mode'] == "CPG":
            plt.figure()
            plt.plot(r_values, label="r")
            plt.plot(rdot_values, label="r_dot")
            plt.plot(theta_values, label="theta")
            plt.plot(theta_dot_values, label="theta_dot")
            plt.xlabel('Time (s)')
            plt.ylabel('CPG States')
            plt.title('CPG States')
        plt.show()
        base_velocity = []
    
    if env.envs[0].env.robot.GetContactInfo()[3][3] == 1:
        time_up_array.append(time_up)
        time_up = 0
        time_down += TIME_STEP
    if env.envs[0].env.robot.GetContactInfo()[3][3] == 0:
        time_down_array.append(time_down)
        time_down = 0
        time_up += TIME_STEP
    foot_positions_flat = env.envs[0].env.robot.GetFootPositions()

    for i in range(4):  # Assuming 4 legs
        # Add foot positions to dictionary
        foot_positions[i]['x'].append(foot_positions_flat[i*3])
        foot_positions[i]['z'].append(env.envs[0].env.robot.GetBasePosition()[2]+foot_positions_flat[i*3+2])
    if env_config['motor_control_mode'] == "CPG":
        r_values.append(env.envs[0].env._cpg.get_r())
        rdot_values.append(env.envs[0].env._cpg.get_dr())
        theta_values.append(env.envs[0].env._cpg.get_theta())
        theta_dot_values.append(env.envs[0].env._cpg.get_dtheta())

    energy += np.abs(np.dot(env.envs[0].env.robot.GetMotorTorques(), env.envs[0].env.robot.GetMotorVelocities()))*env.envs[0].env._time_step
    # [TODO] save data from current robot states for plots 
    # To get base position, for example: env.envs[0].env.robot.GetBasePosition() 
    base_velocity.append(np.linalg.norm(env.envs[0].env.robot.GetBaseLinearVelocity()[0:2]))
    
# [TODO] make plots:


