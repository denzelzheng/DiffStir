import os
import json
import taichi as ti
import numpy as np
import math
import open3d as o3d
import matplotlib.pyplot as plt

ti.reset()
ti.init(arch=ti.cuda, device_memory_GB=20.3,
        kernel_profiler=True, default_fp=ti.f32)

print()
dim = 3
bound = 3
current_dir = os.path.abspath(os.path.dirname(__file__))
config_path = current_dir + '/../../config/'
arguments_file = open(config_path + 'arguments.json')
arguments = json.load(arguments_file)
simulation_setting = arguments['simulation_setting']
dt = simulation_setting['dt']
n_grid = simulation_setting['n_grid']
frames_per_clip = simulation_setting['frames_per_clip']
dx, inv_dx = 1 / n_grid, float(n_grid)
n_particles = simulation_setting['n_model_particles_generated']
n_container_particles = simulation_setting['n_container_model_particles_generated']
n_operator_particles = simulation_setting['n_operator_model_particles_generated']
uniform_scaling_factor = simulation_setting['uniform_scaling_factor']
scene_information = arguments['scene_information']
container_diameter = scene_information['container_diameter']
container_height = scene_information['container_height']
container_radius = container_diameter / 2
container_wall_thickness = scene_information['container_wall_thickness']
liquid_height = scene_information['liquid_height']
operator_radius = scene_information['operator_radius']
operator_length = scene_information['operator_length']
liquid_density = scene_information['liquid_density']
operator_density = scene_information['operator_density']
container_density = scene_information['container_density']
operator_bottom_height = scene_information['operator_bottom_height']
liquid_radius = container_radius - container_wall_thickness
operator_p_vol = operator_radius ** 2 * operator_length * np.pi / n_operator_particles
liquid_p_vol = liquid_radius ** 2 * liquid_height * np.pi / n_particles

operator_p_mass = operator_density * operator_p_vol
liquid_p_mass = liquid_density * liquid_p_vol
area_bottom = np.pi * (container_radius ** 2)
area_wall = container_diameter * 2 * np.pi * container_height
container_area = area_wall + area_bottom
n_particles_container_wall = int(n_container_particles * (area_wall / container_area))
n_particles_container_bottom = n_container_particles - n_particles_container_wall
container_p_vol = (container_area * container_wall_thickness) / n_container_particles
container_p_mass = container_p_vol * container_density
operator_mass = operator_p_mass * n_operator_particles
print("operator_mass", operator_mass)

data_path = current_dir + '/../../data/'
traj_file_name = data_path + 'traj.npy'
operator_trajectory = np.load(traj_file_name)
forces_file_name = data_path + '/force.npy'
force_magnitudes = np.load(forces_file_name)

total_frames = operator_trajectory.shape[0]
print("total_frames", total_frames)
if frames_per_clip >= total_frames:
    frames_per_clip = total_frames - 1
n_clips = int(total_frames / frames_per_clip)
print("n_clips", n_clips)
max_steps = total_frames
n_simulation_frames = max_steps



operator_trajectory = operator_trajectory @ np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
operator_trajectory_center = (np.amax(operator_trajectory, axis=0) - np.amin(operator_trajectory, axis=0)) / 2
operator_trajectory_center += np.amin(operator_trajectory, axis=0)
operator_trajectory = operator_trajectory - operator_trajectory_center
initial_pos_related_to_center = operator_trajectory[0] - operator_trajectory_center
container_particles = []
for i in range(n_particles_container_wall):
    while True:
        x, z = np.random.uniform(-container_radius, container_radius, 2)
        y = np.random.uniform(dx + container_wall_thickness, dx + container_height + container_wall_thickness)
        if (container_radius - container_wall_thickness) ** 2 <= x ** 2 + z ** 2 <= container_radius ** 2:
            container_particles.append((x, y, z))
            break
for i in range(n_particles_container_bottom):
    while True:
        x, z = np.random.uniform(-container_radius, container_radius, 2)
        y = np.random.uniform(dx, dx + container_wall_thickness)
        if x ** 2 + z ** 2 <= container_radius ** 2:
            container_particles.append((x, y, z))
            break
liquid_particles = []
for i in range(n_particles):
    while True:
        x, z = np.random.uniform(-liquid_radius, liquid_radius, 2)
        y = np.random.uniform(2 * dx + container_wall_thickness,
                              liquid_height + container_wall_thickness + 2 * dx)
        if x ** 2 + z ** 2 <= liquid_radius ** 2:
            liquid_particles.append((x, y, z))
            break
operator_particles = []
for i in range(n_operator_particles):
    while True:
        x, z = np.random.uniform(-operator_radius, operator_radius, 2)
        y = np.random.uniform(dx + operator_bottom_height + container_wall_thickness,
                              operator_length + operator_bottom_height + dx
                              + container_wall_thickness)
        if x ** 2 + z ** 2 <= operator_radius ** 2:
            operator_particles.append((x, y, z))
            break
center_placement = np.asarray([0.5, (bound - 1) * dx, 0.5])
container_particles = np.array(container_particles) + center_placement
liquid_particles = np.array(liquid_particles) + center_placement
operator_particles = np.array(operator_particles) + center_placement
operator_particles += initial_pos_related_to_center

pos_series = list()
operator_pos_series = list()
container_pos_series = list()

index = lambda: ti.field(dtype=ti.i32)
scalar = lambda: ti.field(dtype=ti.f32)
vec = lambda: ti.Vector.field(dim, dtype=ti.f32)
mat = lambda: ti.Matrix.field(dim, dim, dtype=ti.f32)

model_index = index()
ti.root.place(model_index)
n_constitutive_parameters = 4
operator_F = mat()
ti.root.dense(ti.k, max_steps).dense(ti.l, n_operator_particles).place(operator_F)
learning_rate = ti.field(dtype=ti.f32, shape=n_constitutive_parameters)
constitutive_parameters = ti.field(dtype=ti.f32, shape=n_constitutive_parameters)
parameters_lower_bound = [0] * n_constitutive_parameters
parameters_upper_bound = [0] * n_constitutive_parameters
grid_v_out = vec()
x, v = vec(), vec()
operator_x, operator_v = vec(), vec()
operator_dv = vec()
container_x, container_v = vec(), vec()
C, F, operator_C, container_C = mat(), mat(), mat(), mat()
loss = scalar()
viscosity_record = scalar()
ti.root.dense(ti.k, max_steps).dense(ti.l, n_particles).place(viscosity_record)
ti.root.place(loss)
ti.root.dense(ti.ijk, n_grid).place(grid_v_out)
initial_x = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)
initial_operator_x = ti.Vector.field(3, dtype=ti.f32, shape=n_operator_particles)
initial_container_x = ti.Vector.field(3, dtype=ti.f32, shape=n_container_particles)
operator_kinematic_seq = ti.Vector.field(3, dtype=ti.f32, shape=n_simulation_frames)
operator_forces_seq = ti.field(dtype=ti.f32, shape=n_simulation_frames)
total_operator_dv = ti.Vector.field(3, dtype=ti.f32, shape=n_simulation_frames)
total_operator_v = ti.Vector.field(3, dtype=ti.f32, shape=n_simulation_frames)
ti.root.dense(ti.k, max_steps).dense(ti.l, n_particles).place(x, v, C, F)
ti.root.dense(ti.k, max_steps).dense(ti.l, n_operator_particles).place(operator_dv)
ti.root.dense(ti.k, max_steps).dense(ti.l, n_operator_particles).place(operator_x, operator_v, operator_C)
ti.root.dense(ti.k, max_steps).dense(ti.l, n_container_particles).place(container_x, container_v, container_C)
grid_v = ti.Vector.field(3, dtype=ti.f32, shape=(n_grid, n_grid, n_grid))  # grid node momentum/velocity
grid_m = ti.field(dtype=ti.f32, shape=(n_grid, n_grid, n_grid))  # grid node mass
F_tmp = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)  # deformation gradient
U = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)
V = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)
sig = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)
ti.root.lazy_grad()