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

""" Run CPG """
import time
import numpy as np
import matplotlib
from tqdm import tqdm

# adapt as needed for your system
# from sys import platform
# if platform =="darwin":
#   matplotlib.use("Qt5Agg")
# else:
#   matplotlib.use('TkAgg')

from matplotlib import pyplot as plt

from env.hopf_network import HopfNetwork
from env.quadruped_gym_env import QuadrupedGymEnv


ADD_CARTESIAN_PD = True
TIME_STEP = 0.001
foot_y = 0.0838 # this is the hip length 
sideSign = np.array([-1, 1, -1, 1]) # get correct hip sign (body right is negative)

env = QuadrupedGymEnv(render=True,              # visualize
                    on_rack=False,              # useful for debugging! 
                    isRLGymInterface=False,     # not using RL
                    time_step=TIME_STEP,
                    action_repeat=1,
                    motor_control_mode="TORQUE",
                    add_noise=False,    # start in ideal conditions
                    # record_video=True
                    )

# initialize Hopf Network, supply gait
cpg = HopfNetwork(time_step=TIME_STEP, 
                  omega_swing = 8*2*np.pi, 
                  omega_stance = 10*2*np.pi,
                  gait = "PACE",
                  mu = 1**2,
                  alpha=50,
                  ground_clearance=0.05,
                  robot_height=0.28,
                  ground_penetration=0.01,
                  coupling_strength=1)

TEST_STEPS = int(2 / (TIME_STEP))
t = np.arange(TEST_STEPS)*TIME_STEP

# [TODO] initialize data structures to save CPG and robot states
r_values = [[] for _ in range(4)]
rdot_values = [[] for _ in range(4)]
theta_values = [[] for _ in range(4)]
theta_dot_values = [[] for _ in range(4)]

x_desired = []
z_desired = []
x_actual = []
z_actual = []
joint_state_actual = [[] for _ in range(3)]
joint_state_desired = [[] for _ in range(3)]


############## Sample Gains
# joint PD gains
kp=np.array([200,50,80])
kd=np.array([2,0.7,0.7])
# Cartesian PD gains
kpCartesian = np.diag([2400]*3)
kdCartesian = np.diag([35]*3)
time_passed = 0
prev_time = 0
energy = 0
base_vel = []
time_up = 0
time_down = 0

for j in tqdm(range(TEST_STEPS)):
  # initialize torque array to send to motors
  action = np.zeros(12) 
  r = cpg.get_r()
  rdot = cpg.get_dr()
  theta = cpg.get_theta()
  theta_dot = cpg.get_dtheta()

# Save the state values
  for i in range(4):
    r_values[i].append(r[i])
    rdot_values[i].append(rdot[i])
    theta_values[i].append(theta[i])
    theta_dot_values[i].append(theta_dot[i])
  # get desired foot positions from CPG 
  xs,zs = cpg.update()
  # [TODO] get current motor angles and velocities for joint PD, see GetMotorAngles(), GetMotorVelocities() in quadruped.py
  q = env.robot.GetMotorAngles()
  dq = env.robot.GetMotorVelocities()
  # loop through desired foot positions and calculate torques
  for i in range(4):
    # initialize torques for legi
    tau = np.zeros(3)
    # get desired foot i pos (xi, yi, zi) in leg frame
    leg_xyz = np.array([xs[i],sideSign[i] * foot_y,zs[i]])
    if i == 1:
      x_desired.append(xs[1])
      z_desired.append(zs[1])
      x_actual.append(env.robot.ComputeJacobianAndPosition(1)[1][0])
      z_actual.append(env.robot.ComputeJacobianAndPosition(1)[1][2])
      for k in range(3):
        joint_state_actual[k].append(env.robot.GetMotorAngles()[k])
    # call inverse kinematics to get corresponding joint angles (see ComputeInverseKinematics() in quadruped.py)

    leg_q = env.robot.ComputeInverseKinematics(i, leg_xyz) # [TODO]
    if i == 1:
      for k in range(3):
        joint_state_desired[k].append(leg_q[k])
    # Add joint PD contribution to tau for leg i (Equation 4)

    tau += kp*(leg_q - q[3*i:3*(i+1)]) + kd*(-dq[3*i:3*(i+1)]) # [TODO] 

    # add Cartesian PD contribution
    if ADD_CARTESIAN_PD:
      # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quadruped.py)

      # [TODO] 
      J, pos = env.robot.ComputeJacobianAndPosition(i)

      # Get current foot velocity in leg frame (Equation 2)
      # [TODO] 
      vel = J@dq[3*i:3*(i+1)]
      # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
      tau += np.transpose(J)@(kpCartesian@(leg_xyz - pos) + kdCartesian@(-vel))  # [TODO]

    # Set tau for legi in action vector
    action[3*i:3*i+3] = tau
  # calculate the energy with the actions
  energy += np.abs(np.dot(action, dq))*TIME_STEP
  base_vel.append(env.robot.GetBaseLinearVelocity()[0])
  # calculate the time duration of one step
  
  if env.robot.GetContactInfo()[3][0] == 1:
    time_up = 0
    time_down += TIME_STEP
  if env.robot.GetContactInfo()[3][0] == 0:
    time_down = 0
    time_up += TIME_STEP
  print("Stance Time: ", time_down,"Swing Time: " ,time_up)
  # send torques to robot and simulate TIME_STEP seconds 
  env.step(action) 
  # [TODO] save any CPG or robot states


base_pos = env.robot.GetBasePosition()[0]
COT = energy / (base_pos * 9.81 * 12)
print("base_pos: ", base_pos)
print("Cost of Transport: ", COT)
print("Average Velocity: ", np.mean(base_vel))

##################################################### 
# PLOTS
#####################################################
# Plot the CPG states

# plot the CPG states for just one leg in 4x1 subplots

max_val = [np.max(r_values), np.max(rdot_values), np.max(theta_values), np.max(theta_dot_values)]
max_val = np.ceil(max_val)

plt.figure()

plt.subplot(411)
plt.plot(t, r_values[0], label='r')
plt.ylabel('r', rotation = 0, labelpad=10)
ax = plt.gca()
ax.set_yticks(np.arange(0, max_val[0]+0.1, max_val[0]))
plt.xticks([])  # remove x-ticks

plt.subplot(412)
plt.plot(t, theta_values[0], label='$\\theta$')
plt.ylabel('$\\theta$', rotation = 0, labelpad=10)
ax = plt.gca()
ax.set_yticks(np.arange(0, max_val[2]+0.1, max_val[2]))
plt.xticks([])  # remove x-ticks

plt.subplot(413)
plt.plot(t, rdot_values[1], label='$\\dot{r}$')
plt.ylabel('$\\dot{r}$', rotation = 0, labelpad=10)
ax = plt.gca()
ax.set_yticks(np.arange(0, max_val[1]+0.1, max_val[1]))
plt.xticks([])  # remove x-ticks

plt.subplot(414)
plt.plot(t, theta_dot_values[1], label='$\\dot{\\theta}$')
plt.ylabel('$\\dot{\\theta}$', rotation = 0, labelpad=10)
ax = plt.gca()
ax.set_yticks(np.arange(0, max_val[3]+0.1, max_val[3]))

plt.xlabel('t [s]')
plt.savefig('cpg_states.pdf', format='pdf')

# plt.show()

# Plot the actual and desired foot positions
plt.figure()
plt.plot(x_desired, z_desired, label='Desired')
plt.plot(x_actual, z_actual, label='Actual')
plt.legend()
plt.xlabel('x [m]')
plt.ylabel('z [m]')
plt.savefig('foot_pos_limit_cycle.pdf', format='pdf')


# plot x and z seperately against time
plt.figure()
plt.plot(t, x_desired, label='X Desired')
plt.plot(t, x_actual, label='X Actual')
plt.plot(t, z_desired, label='Z Desired')
plt.plot(t, z_actual, label='Z Actual')
plt.legend()
plt.xlabel('t [s]')
plt.ylabel('x, z [m]')
plt.savefig('foot_xz_cartesian.pdf', format='pdf')


# plot all joint angles with desired joint angles in just one graph
plt.figure()
plt.plot(t, joint_state_desired[0], label='Hip Desired')
plt.plot(t, joint_state_actual[0], label='Hip Actual')
plt.plot(t, joint_state_desired[1], label='Thigh Desired')
plt.plot(t, joint_state_actual[1], label='Thigh Actual')
plt.plot(t, joint_state_desired[2], label='Calf Desired')
plt.plot(t, joint_state_actual[2], label='Calf Actual')
plt.legend()
plt.xlabel('t [s]')
plt.ylabel('Joint angles [rad]')
plt.savefig('joint_angles_cartesian.pdf', format='pdf')
plt.show()







