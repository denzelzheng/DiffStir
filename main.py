from src.gradient_descent_module import *
from src.force_calculation_and_animation_module import *

initialize_objects()
constitutive_parameters_config_file = os.path.abspath(os.path.dirname(__file__)) + "/config/constitutive_parameters_config.npy"
constitutive_parameters_config = np.load(constitutive_parameters_config_file)
loss_for_models = []
optimized_parameters_for_models = []
for i in range(3):
    model_index[None] = i
    for j in range(4):
        constitutive_parameters[i] = constitutive_parameters_config[0, i, j]
        parameters_lower_bound[i] = constitutive_parameters_config[1, i, j]
        parameters_upper_bound[i] = constitutive_parameters_config[2, i, j]
    # force_calculation_and_animation((max_steps - 1) * dt)
    optimized_parameters_for_models.append(parameters_fitting(35))
    loss_for_models.append(loss[None])

data = {
    "parameters": optimized_parameters_for_models,
    "losses": loss_for_models
}
with open(os.path.abspath(os.path.dirname(__file__)) + "/output/parameters_and_losses_result.json", "w") as file:
    json.dump(data, file, indent=4)

best_model_index = loss_for_models.index(min(loss_for_models))
for i in range(4):
    constitutive_parameters[i] = optimized_parameters_for_models[best_model_index][i]
model_index[None] = best_model_index
force_calculation_and_animation((max_steps - 1) * dt)
