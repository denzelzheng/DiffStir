from src.simulator.initialization import *

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
        if model_index[None] == 0:
            eta0 = constitutive_parameters[0]
            eta_inf = constitutive_parameters[1]
            n_carreau = constitutive_parameters[2]
            lambda_carreau = constitutive_parameters[3]
            eta = eta_inf + (eta0 - eta_inf) * ((1 + (lambda_carreau * gamma) ** 2) ** ((n_carreau - 1) / 2))
            strain_rate = 0.5 * (F_dot + F_dot.transpose())

        # cross
        if model_index[None] == 1:
            eta0 = constitutive_parameters[0]
            eta_inf = constitutive_parameters[1]
            n_cross = constitutive_parameters[2]
            lambda_cross = constitutive_parameters[3]
            eta = eta_inf + (eta0 - eta_inf) * (1 / (1 + (lambda_cross * gamma) ** (n_cross -1)))
            strain_rate = 0.5 * (F_dot + F_dot.transpose())

        # Herschel-Bulkley
        if model_index[None] == 2:
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