import torch

def get_step_lr_pt(optimizer, mode='min', patience=10, factor=0.1):
    """
    Create a ReduceLROnPlateau learning rate scheduler for the provided optimizer.

    :param optimizer: The optimizer for which to create the scheduler.
    :param mode: One of 'min' or 'max'. 'min' will reduce the LR when the quantity monitored has stopped decreasing; 'max' will reduce it when the quantity monitored has stopped increasing.
    :param patience: Number of epochs with no improvement after which learning rate will be reduced.
    :param factor: Factor by which the learning rate will be reduced. new_lr = lr * factor.
    :return: A ReduceLROnPlateau learning rate scheduler.
    """
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, patience=patience, factor=factor)
    return scheduler
