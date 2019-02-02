

def update_target_weights_hard(target_variables, model_variables):
    for target, current in zip(target_variables, model_variables):
        target.assign(current)


def update_target_weights_soft(target_variables, model_variables, beta):
    for target, current in zip(target_variables, model_variables):
        target.assign_add(beta * (current - target))