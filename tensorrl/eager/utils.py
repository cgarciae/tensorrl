def update_variables(variables, values):
    for variable, value in zip(variables, values):
        variable.assign(value)


def update_variables_soft(variables, values, beta):
    for variable, value in zip(variables, values):
        variable.assign_add(beta * (value - variable))