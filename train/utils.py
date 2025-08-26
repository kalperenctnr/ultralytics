def generate_layer_lrs(model, base_lr=1e-3, split_layer=10, factor=0.1):
    """
    Generate a dictionary mapping parameter names to learning rates.
    Layers up to `split_layer` get reduced LR, others get base LR.

    Args:
        model (torch.nn.Module): Model to extract parameter names from.
        base_lr (float): Base learning rate.
        split_layer (int): Layer index threshold (inclusive).
        factor (float): Scaling factor for early layers (e.g., 0.1 means lr/10).

    Returns:
        dict: Mapping of parameter names to learning rates.
    """
    layer_lrs = {}
    for idx, (name, param) in enumerate(model.named_parameters()):
        if not param.requires_grad:
            continue

        # Decide LR based on layer index
        if idx <= split_layer:
            lr = base_lr * factor
        else:
            lr = base_lr

        layer_lrs[name] = lr

    return layer_lrs
