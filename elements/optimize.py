import torch

def get_adam_optimizer_pt(model, learning_rate):
    """
    Create an Adam optimizer for the provided model with the specified learning rate.

    :param model: The model for which to create the optimizer.
    :param learning_rate: The learning rate to use with the Adam optimizer.
    :return: An Adam optimizer configured with the model's parameters and the given learning rate.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return optimizer

