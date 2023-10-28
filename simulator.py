import taichi as ti
import numpy as np
import math
import os
import json
import open3d as o3d
import matplotlib.pyplot as plt
import imageio

ti.reset()
ti.init(arch=ti.cuda, device_memory_GB=20.3,
        kernel_profiler=True, default_fp=ti.f32)

print()
dim = 3
bound = 3
current_dir = os.path.abspath(os.path.dirname(__file__))
arguments_file = open(current_dir + '/liquid_arguments.json')
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

traj_file_name = current_dir + '/traj.npy'
operator_trajectory = np.load(traj_file_name)
forces_file_name = current_dir + '/force.npy'
force_magnitudes = np.load(forces_file_name)

total_frames = operator_trajectory.shape[0]
print("total_frames", total_frames)
if frames_per_clip >= total_frames:
    frames_per_clip = total_frames - 1
n_clips = int(total_frames / frames_per_clip)
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




@ti.kernel
def clear_grid():
    for i, j, k in grid_m:
        grid_v[i, j, k] = [0, 0, 0]
        grid_m[i, j, k] = 0
        grid_v.grad[i, j, k] = [0, 0, 0]
        grid_m.grad[i, j, k] = 0
        grid_v_out.grad[i, j, k] = [0, 0, 0]

@ti.kernel
def clear_grad():
        for j in range(max_steps):
            for i in range(n_particles):
                x.grad[j, i] = [0, 0, 0]
                v.grad[j, i] = [0, 0, 0]
                F.grad[j, i] = ti.Matrix([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
                C.grad[j, i] = ti.Matrix([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
            for i in range(n_operator_particles):
                operator_x.grad[j, i] = [0, 0, 0]
                operator_v.grad[j, i] = [0, 0, 0]
                operator_C.grad[j, i] = ti.Matrix([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
            for i in range(n_container_particles):
                container_x.grad[j, i] = [0, 0, 0]
                container_v.grad[j, i] = [0, 0, 0]
                container_C.grad[j, i] = ti.Matrix([[0, 0, 0], [0, 1, 0], [0, 0, 0]])



def zero_vec():
    return [0.0, 0.0, 0.0]


def zero_matrix():
    return [zero_vec(), zero_vec(), zero_vec()]


@ti.kernel
def compute_F_tmp(f: ti.i32):
    for p in range(0, n_particles):  # Particle state update and scatter to grid (P2G)
        F_tmp[p] = (ti.Matrix.identity(ti.f32, dim) + dt * C[f, p]) @ F[f, p]


@ti.kernel
def clear_SVD_grad():
    zero = ti.Matrix.zero(ti.f32, dim, dim)
    for i in range(0, n_particles):
        U.grad[i] = zero
        sig.grad[i] = zero
        V.grad[i] = zero
        F_tmp.grad[i] = zero


@ti.kernel
def svd():
    for p in range(0, n_particles):
        U[p], sig[p], V[p] = ti.svd(F_tmp[p])


@ti.func
def clamp(a):
    if a >= 0:
        a = max(a, 1e-6)
    else:
        a = min(a, -1e-6)
    return a


@ti.func
def backward_svd(gu, gsigma, gv, u_matrix, sigma, v_matrix):
    # https://github.com/pytorch/pytorch/blob/ab0a04dc9c8b84d4a03412f1c21a6c4a2cefd36c/tools/autograd/templates/Functions.cpp
    vt = v_matrix.transpose()
    ut = u_matrix.transpose()
    sigma_term = u_matrix @ gsigma @ vt

    s = ti.Vector.zero(ti.f32, dim)
    s = ti.Vector([sigma[0, 0], sigma[1, 1], sigma[2, 2]]) ** 2
    f = ti.Matrix.zero(ti.f32, dim, dim)
    for i, j in ti.static(ti.ndrange(dim, dim)):
        if i == j:
            f[i, j] = 0
        else:
            f[i, j] = 1. / clamp(s[j] - s[i])
    u_term = u_matrix @ ((f * (ut @ gu - gu.transpose() @ u_matrix)) @ sigma) @ vt
    v_term = u_matrix @ (sigma @ ((f * (vt @ gv - gv.transpose() @ v_matrix)) @ vt))
    return u_term + v_term + sigma_term


@ti.kernel
def svd_grad():
    for p in range(0, n_particles):
        F_tmp.grad[p] += backward_svd(U.grad[p], sig.grad[p], V.grad[p], U[p], sig[p], V[p])

@ti.func
def make_matrix_from_diag(d):
    return ti.Matrix([[d[0], 0.0, 0.0], [0.0, d[1], 0.0], [0.0, 0.0, d[2]]], dt=ti.f32)

@ti.func
def norm(x, eps=1e-8):
    return ti.sqrt(x.dot(x) + eps)


@ti.kernel
def p2g(f: ti.i32):
    for p in range(0, n_particles):

        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, ti.i32)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_F = F_tmp[p]
        k = 20000
        J = new_F.determinant()
        F_dot = C[f, p]
        gamma = C[f, p].norm() + 1e-6
        eta = 0
        strain_rate = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        # carreau
        if model_index == 0:
            eta0 = constitutive_parameters[0]
            eta_inf = constitutive_parameters[1]
            n_carreau = constitutive_parameters[2]
            lambda_carreau = constitutive_parameters[3]
            eta = eta_inf + (eta0 - eta_inf) * ((1 + (lambda_carreau * gamma) ** 2) ** ((n_carreau - 1) / 2))
            strain_rate = 0.5 * (F_dot + F_dot.transpose())

        # cross
        if model_index == 1:
            eta0 = constitutive_parameters[0]
            eta_inf = constitutive_parameters[1]
            n_cross = constitutive_parameters[2]
            lambda_cross = constitutive_parameters[3]
            eta = eta_inf + (eta0 - eta_inf) * (1 / (1 + (lambda_cross * gamma) ** (n_cross -1)))
            strain_rate = 0.5 * (F_dot + F_dot.transpose())

        # Herschel-Bulkley
        if model_index == 2:
            K = constitutive_parameters[0]
            tau_y = constitutive_parameters[1]
            n_hb = constitutive_parameters[2]
            delta_eta = K * (gamma ** (n_hb - 1))
            eta = tau_y / gamma
            if delta_eta > 0:
                eta += delta_eta
            strain_rate = 0.5 * (F_dot + F_dot.transpose())

        stress = eta * strain_rate + \
                 ti.Matrix.identity(float, 3) * (k / 2) * (J - (1 / J) - 1) * J
        stress = (-dt * liquid_p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + liquid_p_mass * C[f, p]
        temp_v = v[f, p] + [0, dt * uniform_scaling_factor * (-9.81), 0]
        F[f + 1, p] = new_F

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    offset = ti.Vector([i, j, k])
                    dpos = (ti.cast(ti.Vector([i, j, k]), ti.f32) - fx) * dx
                    weight = w[i](0) * w[j](1) * w[k](2)
                    ti.atomic_add(grid_v[base + offset],
                                  weight * (liquid_p_mass * temp_v + affine @ dpos))
                    ti.atomic_add(grid_m[base + offset], weight * liquid_p_mass)

    for p in range(0, n_operator_particles):
        base = ti.cast(operator_x[f, p] * inv_dx - 0.5, ti.i32)
        fx = operator_x[f, p] * inv_dx - ti.cast(base, ti.i32)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        affine = operator_p_mass * operator_C[f, p]
        temp_v = operator_v[f, p]

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    offset = ti.Vector([i, j, k])
                    dpos = (ti.cast(ti.Vector([i, j, k]), ti.f32) - fx) * dx
                    weight = w[i](0) * w[j](1) * w[k](2)
                    ti.atomic_add(grid_v[base + offset],
                                  weight * (operator_p_mass * temp_v + affine @ dpos))
                    ti.atomic_add(grid_m[base + offset], weight * operator_p_mass)

    for p in range(0, n_container_particles):
        base = ti.cast(container_x[f, p] * inv_dx - 0.5, ti.i32)
        fx = container_x[f, p] * inv_dx - ti.cast(base, ti.i32)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        affine = container_p_mass * container_C[f, p]

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    offset = ti.Vector([i, j, k])
                    dpos = (ti.cast(ti.Vector([i, j, k]), ti.f32) - fx) * dx
                    weight = w[i](0) * w[j](1) * w[k](2)
                    ti.atomic_add(grid_v[base + offset],
                                  weight * (container_p_mass * container_v[f, p] + affine @ dpos))
                    ti.atomic_add(grid_m[base + offset], weight * container_p_mass)


@ti.kernel
def grid_op(f: ti.i32):
    for i, j, k in grid_m:
        inv_m = 1 / (grid_m[i, j, k] + 1e-10)
        v_out = inv_m * grid_v[i, j, k]
        t = f * dt
        r = 0.0415 - 0.5 * dx
        cy, cz = 0.5, 0.5
        dist = ti.Vector([i * dx - cy, 0, k * dx - cz])
        if dist.norm_sqr() > r**2:
            dist = dist.normalized()
            v_out -= dist * v_out.dot(dist)
        if i < bound and v_out[0] < 0:
            v_out[0] = 0
        if i > n_grid - bound and v_out[0] > 0:
            v_out[0] = 0
        if k < bound and v_out[2] < 0:
            v_out[2] = 0
        if k > n_grid - bound and v_out[2] > 0:
            v_out[2] = 0
        if j < bound and v_out[1] < 0:
            v_out[1] = 0
        if j > n_grid - bound and v_out[1] > 0:
            v_out[0] = 0
            v_out[1] = 0
            v_out[2] = 0
        grid_v_out[i, j, k] = v_out


@ti.kernel
def g2p(f: ti.i32):
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, ti.f32)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector([0.0, 0.0, 0.0])
        new_C = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    dpos = ti.cast(ti.Vector([i, j, k]), ti.f32) - fx
                    g_v = grid_v_out[base(0) + i, base(1) + j, base(2) + k]
                    weight = w[i](0) * w[j](1) * w[k](2)
                    new_v += weight * g_v
                    new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx

        v[f + 1, p] = new_v
        x[f + 1, p] = x[f, p] + dt * v[f + 1, p]
        C[f + 1, p] = new_C

    for p in range(n_operator_particles):
        base = ti.cast(operator_x[f, p] * inv_dx - 0.5, ti.i32)
        fx = operator_x[f, p] * inv_dx - ti.cast(base, ti.f32)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_operator_v = ti.Vector([0.0, 0.0, 0.0])
        new_operator_C = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    dpos = ti.cast(ti.Vector([i, j, k]), ti.f32) - fx
                    g_v = grid_v_out[base(0) + i, base(1) + j, base(2) + k]
                    weight = w[i](0) * w[j](1) * w[k](2)
                    new_operator_v += weight * g_v
                    new_operator_C += 4 * weight * g_v.outer_product(dpos) * inv_dx

        temp_v = (operator_kinematic_seq[f + 1] - operator_kinematic_seq[f]) / dt
        operator_dv[f + 1, p] = temp_v - new_operator_v

        operator_v[f + 1, p] = temp_v
        operator_x[f + 1, p] = operator_x[f, p] + temp_v * dt
        operator_C[f + 1, p] = new_operator_C

    for p in range(n_container_particles):
        base = ti.cast(container_x[f, p] * inv_dx - 0.5, ti.i32)
        fx = container_x[f, p] * inv_dx - ti.cast(base, ti.f32)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_container_v = ti.Vector([0.0, 0.0, 0.0])
        new_container_C = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    dpos = ti.cast(ti.Vector([i, j, k]), ti.f32) - fx
                    g_v = grid_v_out[base(0) + i, base(1) + j, base(2) + k]
                    weight = w[i](0) * w[j](1) * w[k](2)
                    new_container_v += weight * g_v
                    new_container_C += 4 * weight * g_v.outer_product(dpos) * inv_dx

        container_v[f + 1, p] = [0, 0, 0]
        container_x[f + 1, p] = container_x[f, p]
        container_C[f + 1, p] = new_container_C


def initialize_objects():
    initial_x.from_numpy(liquid_particles)
    initial_operator_x.from_numpy(operator_particles)
    initial_container_x.from_numpy(container_particles)
    operator_kinematic_seq.from_numpy(operator_trajectory)
    initialize_ti_field()


@ti.kernel
def initialize_ti_field():
    for i in range(max_steps):
        total_operator_dv[i] = [0, 0, 0]
        total_operator_v[i] = [0, 0, 0]
    for i in range(n_particles):
        x[0, i] = initial_x[i]
        v[0, i] = [0, 0, 0]
        F[0, i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        C[0, i] = ti.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    for i in range(n_operator_particles):
        operator_x[0, i] = initial_operator_x[i]
        operator_v[0, i] = [0, 0, 0]
        # operator_F[0, i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        operator_C[0, i] = ti.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    for i in range(n_container_particles):
        container_x[0, i] = initial_container_x[i]
        container_v[0, i] = [0, 0, 0]
        container_C[0, i] = ti.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])


def compute_total_force():
    operator_dv_array = operator_dv.to_numpy() * operator_p_mass
    operator_v_array = operator_v.to_numpy() * operator_p_mass
    operator_dv_array = np.sum(operator_dv_array, axis=1)
    operator_v_array = np.sum(operator_v_array, axis=1)

    print((operator_dv_array - operator_v_array) / dt)


@ti.kernel
def compute_loss():
    for step in range(max_steps - 1):
        for p in range(n_operator_particles):
            ti.atomic_add(total_operator_dv[step], operator_dv[step, p])
        operator_acc = total_operator_dv[step] / dt
        operator_force = operator_acc * operator_p_vol * operator_density
        operator_force_magnitudes = operator_force[0]
        diff = abs(operator_force_magnitudes - operator_forces_seq[step])
        loss[None] += diff / (max_steps - 1)

@ti.ad.grad_replaced
def forward(s):
    clear_grid()
    compute_F_tmp(s)
    svd()
    p2g(s)
    grid_op(s)
    g2p(s)


@ti.ad.grad_for(forward)
def backward(s):
    clear_grid()
    clear_SVD_grad()
    compute_F_tmp(s)
    svd()
    p2g(s)
    grid_op(s)
    g2p(s)

    g2p.grad(s)
    grid_op.grad(s)
    p2g.grad(s)
    svd_grad()
    compute_F_tmp.grad(s)


def animation(presentation_time: ti.f32):
    print()
    initialize_objects()
    print("Simulating the process for visualization...")
    loss[None] = 0
    for f in range(int(presentation_time / dt - 1)):
        forward(f)
    compute_loss()
    print("eta0:", constitutive_parameters[0])
    print("Loss:", loss[None])
    pos_series = x.to_numpy()
    operator_pos_series = operator_x.to_numpy()
    container_pos_series = container_x.to_numpy()
    total_operator_dv_series = total_operator_dv.to_numpy() \
                               * operator_p_vol * operator_density / dt
    force_x = total_operator_dv_series[:, 0]
    force_z = total_operator_dv_series[:, 2]
    np.save(current_dir + "/../../result_data/sim_curve_x.npy", force_x)
    np.save(current_dir + "/../../result_data/sim_curve_z.npy", force_z)
    counter = 0
    while True:
        counter += 1
        word = input("Finished. Press p + Enter to continue...")
        if word == 'p':
            break
    pcd = o3d.geometry.PointCloud()
    operator_pcd = o3d.geometry.PointCloud()
    container_pcd = o3d.geometry.PointCloud()
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    elevation = dx * (bound - 1)
    lower_bound = elevation
    upper_bound = 1 - elevation
    points = [
        [lower_bound, lower_bound, lower_bound],
        [upper_bound, lower_bound, lower_bound],
        [lower_bound, upper_bound, lower_bound],
        [upper_bound, upper_bound, lower_bound],
        [lower_bound, lower_bound, upper_bound],
        [upper_bound, lower_bound, upper_bound],
        [lower_bound, upper_bound, upper_bound],
        [upper_bound, upper_bound, upper_bound],
    ]
    lines = [
        [0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6],
        [5, 7], [6, 7], [0, 4], [1, 5], [2, 6], [3, 7],
    ]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    stick_center_series = np.mean(operator_pos_series, axis=1)
    np.save(current_dir + "/liquid_point_clouds.npy", pos_series)
    np.save(current_dir + "/stick_positions.npy", stick_center_series[:-1,:])
    for j in range(f):
        play_rate = 11
        if (j % play_rate == 0):
            pcd.points = o3d.utility.Vector3dVector(pos_series[j])
            operator_pcd.points = o3d.utility.Vector3dVector(operator_pos_series[j])
            operator_pcd.paint_uniform_color([x / 255 for x in [227, 168, 105]])
            operator_pcd, _ = operator_pcd.remove_statistical_outlier(nb_neighbors=3, std_ratio=37.0)
            container_pcd.points = o3d.utility.Vector3dVector(container_pos_series[j])
            container_pcd.paint_uniform_color([x / 255 for x in [255, 121, 111]])
            container_pcd, _ = container_pcd.remove_statistical_outlier(nb_neighbors=3, std_ratio=37.0)
            coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=(0., 0., 0.))

            vis.clear_geometries()
            vis.add_geometry(pcd)
            vis.add_geometry(operator_pcd)
            vis.add_geometry(container_pcd)
            vis.add_geometry(line_set)
            vis.add_geometry(coordinate)
            for i in range(1):
                vis.poll_events()
                vis.update_renderer()
    vis.run()


def magnitude_level(a):
    abs_a = ti.max(abs(a), 1e-11)
    log = math.log10(abs_a)
    if np.isnan(log):
        log = 0
    level = math.floor(log)
    return level


def single_parameter_fitting1(parameter_grad, grad_shock, parameter_index, epochs, epoch_index):
    period = epochs - epoch_index
    parameter = constitutive_parameters[parameter_index]
    lower_bound = parameters_lower_bound[parameter_index]
    upper_bound = parameters_upper_bound[parameter_index]
    lr = learning_rate[parameter_index]
    base = 3
    if abs(parameter_grad) > 0:
        delta = lr * parameter_grad
        if parameter_grad < 0:
            new_magnitude = magnitude_level((upper_bound - parameter) / period)
        if parameter_grad > 0:
            new_magnitude = magnitude_level((lower_bound - parameter) / period)
        if epoch_index < 1:
            lr *= pow(base, new_magnitude - magnitude_level(delta))
            delta = lr * parameter_grad
        parameter_update = parameter - delta
        if lower_bound < parameter_update < upper_bound:
            parameter = parameter_update
            lr *= pow(base, new_magnitude - magnitude_level(delta))
        else:
            if parameter_update < lower_bound:
                new_magnitude = magnitude_level((upper_bound - parameter) / period)
            if parameter_update > upper_bound:
                new_magnitude = magnitude_level((lower_bound - parameter) / period)
            lr *= pow(base, new_magnitude - magnitude_level(delta))
            parameter_update = parameter - lr * parameter_grad
            while parameter_update > upper_bound or parameter_update < lower_bound:
                lr *= 0.5
                parameter_update = parameter - lr * parameter_grad
            parameter = parameter_update
    return parameter, lr


def single_parameter_fitting(parameter_grad, grad_shock, parameter_index, epochs, epoch_index):
    period = epochs - epoch_index
    parameter = constitutive_parameters[parameter_index]
    lower_bound = parameters_lower_bound[parameter_index]
    upper_bound = parameters_upper_bound[parameter_index]
    lr = learning_rate[parameter_index]
    base = 3
    if np.isnan((upper_bound - parameter) / period):
        print("(upper_bound - parameter) / period) is nan", upper_bound, parameter)
    if np.isnan((lower_bound - parameter) / period):
        print("(lower_bound - parameter) / period) is nan", lower_bound, parameter)
    level_to_top = magnitude_level((upper_bound - parameter) / period)
    level_to_bottom = magnitude_level((lower_bound - parameter) / period)
    if not np.isnan(parameter_grad) and abs(parameter_grad) > 1e-31:
        delta = lr * parameter_grad
        lr_to_top = lr * pow(base, level_to_top - magnitude_level(delta))
        lr_to_bottom = lr * pow(base, level_to_bottom - magnitude_level(delta))
        max_lr = max(lr_to_top, lr_to_bottom)
        min_lr = min(lr_to_top, lr_to_bottom)

        lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(epoch_index / epochs))
        parameter_update = parameter - lr * parameter_grad
        if lower_bound < parameter_update < upper_bound:
            parameter = parameter_update

        else:
            new_magnitude = magnitude_level(delta)
            if parameter_update <= lower_bound:
                new_magnitude = level_to_top
            if parameter_update >= upper_bound:
                new_magnitude = level_to_bottom
            lr *= pow(base, new_magnitude - magnitude_level(delta))
            delta = lr * parameter_grad
            parameter_update = parameter - delta
            while parameter_update > upper_bound or parameter_update < lower_bound:
                lr *= 0.5
                parameter_update = parameter - lr * parameter_grad
            parameter = parameter_update
    return parameter, lr


def parameters_fitting(epoch_setting: ti.i32):
    initialize_objects()
    for clip in range(n_clips):
        grad_list = [0] * n_constitutive_parameters
        for lr in range(n_constitutive_parameters):
            learning_rate[lr] = 1
        grad_shock = [0] * n_constitutive_parameters
        loss_list =[]
        parameter_lists = []
        for i in range(4):
            parameter_lists.append([])
        initialize_ti_field()
        epoch = epoch_setting
        if (clip < 1):
            epoch = epoch_setting * 3
        for i in range(epoch):
            print()
            print("clip: ", clip, "|epoch: ", i)
            loss[None] = 0
            with ti.ad.Tape(loss=loss):
                loss[None] = 0
                for f in range(max_steps - 1):
                    forward(f)
                compute_loss()
            loss_list.append(loss[None])
            for j in range(4):
                parameter_lists[j].append(constitutive_parameters[j])
            print('>>> loss =', loss[None])
            for j in range(n_constitutive_parameters):
                if grad_list[j] * constitutive_parameters.grad[j] < 0:
                    grad_shock[j] = 1
                grad_list[j] = constitutive_parameters.grad[j]
            for j in range(n_constitutive_parameters):
                constitutive_parameters[j], learning_rate[j] = single_parameter_fitting1(grad_list[j], grad_shock[j],
                                                                                         j, epoch, i)
                print("updated parameter:", constitutive_parameters[j])
                print("parameters grad:", constitutive_parameters.grad[j])
            print()
            initialize_ti_field()

        optimized_parameters = []
        best_epoch = loss_list.index(min(loss_list))
        for i in range(4):
            constitutive_parameters[i] = parameter_lists[i][best_epoch]
            optimized_parameters.append(constitutive_parameters[i])
        loss[None] = loss_list[best_epoch]
        print("loss of clip:", loss[None])
        initial_operator_x.from_numpy(operator_x.to_numpy()[max_steps - 1])
        initial_x.from_numpy(x.to_numpy()[max_steps - 1])
        if (clip < n_clips - 1):
            start = clip * frames_per_clip
            end = (clip + 1) * frames_per_clip
            operator_kinematic_seq.from_numpy(operator_trajectory[start: end, :])
            operator_forces_seq.from_numpy(force_magnitudes[start: end])
            initialize_ti_field()
        return optimized_parameters

def train(epoch: ti.i32):
    E_grad, nu_grad, yield_stress_grad, viscosity_v_grad, viscosity_d_grad = 0, 0, 0, 0, 0
    initialize_objects()
    grad_list = [0] * n_constitutive_parameters
    print()
    loss_list, parameter0_list = [], []
    for lr in range(n_constitutive_parameters):
        learning_rate[lr] = 1
    grad_shock = [0] * n_constitutive_parameters
    for i in range(epoch):
        print()
        print("epoch: ", i)
        loss[None] = 0
        with ti.ad.Tape(loss=loss):
            loss[None] = 0
            for f in range(max_steps - 1):
                forward(f)
            compute_loss()

        loss_list.append(loss[None])
        parameter0_list.append(constitutive_parameters[0])
        print()
        print('>>> loss =', loss[None])
        print()
        for j in range(n_constitutive_parameters):
            if grad_list[j] * constitutive_parameters.grad[j] < 0:
                grad_shock[j] = 1
            grad_list[j] = constitutive_parameters.grad[j]
        initialize_objects()
        for j in range(n_constitutive_parameters):
            constitutive_parameters[j], learning_rate[j] = single_parameter_fitting1(grad_list[j], grad_shock[j],
                                                                                     j, epoch, i)
            print("updated parameter:", constitutive_parameters[j])
            print("parameters grad:", constitutive_parameters.grad[j])
        print()

    best_epoch = loss_list.index(min(loss_list))

    constitutive_parameters[0] = parameter0_list[best_epoch]
    loss[None] = loss_list[best_epoch]

    initial_dynamics_viscosity_array = np.array(parameter0_list)
    loss_array = np.array(loss_list)

    plt.plot(initial_dynamics_viscosity_array, loss_array)
    plt.xlabel('initial_dynamics_viscosity-axis')
    plt.ylabel('loss-axis')
    plt.title('initial_dynamics_viscosity - loss Plot')
    plt.grid(True)
    print("Result:")
    print("  min loss is", loss[None])
    print("  prediction eta0 is", constitutive_parameters[0])
    print()

initialize_objects()

constitutive_parameters_config_file = current_dir + "/constitutive_parameters_config.npy"
constitutive_parameters_config = np.load(constitutive_parameters_config_file)
loss_for_models = []
optimized_parameters_for_models = []
for model_index in range(3):
    for i in range(4):
        constitutive_parameters[i] = constitutive_parameters_config[0, model_index, i]
        parameters_lower_bound[i] = constitutive_parameters_config[1, model_index, i]
        parameters_upper_bound[i] = constitutive_parameters_config[2, model_index, i]
    optimized_parameters_for_models.append(parameters_fitting(35))
    loss_for_models.append(loss[None])

data = {
    "parameters": optimized_parameters_for_models,
    "losses": loss_for_models
}
with open(current_dir + "/parameters_and_losses_result.json", "w") as file:
    json.dump(data, file, indent=4)

best_model_index = loss_for_models.index(min(loss_for_models))
for i in range(4):
    constitutive_parameters[i] = optimized_parameters_for_models[best_model_index][i]
animation((max_steps - 1) * dt)