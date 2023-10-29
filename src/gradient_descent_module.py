from src.simulator.simulation_functions import *

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