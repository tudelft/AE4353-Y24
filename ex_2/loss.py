import torch.nn.functional as F
import torch

def cross_entropy(output_logits, target_angle, target_vector, smoothing=0.98):
    b, n = output_logits.shape

    # for each node, calculate the difference in angle to the target angle
    target_node_angle_diff = torch.abs(
        torch.linspace(0, 360, n).unsqueeze(0).to(target_angle.device)
        - target_angle.unsqueeze(1)
    )
    # wrap around a full rotation
    target_node_angle_diff = torch.min(
        target_node_angle_diff, 360 - target_node_angle_diff
    )

    exp_encoding = smoothing ** (target_node_angle_diff * n / 360)
    exp_encoding = exp_encoding / exp_encoding.sum()  # normalize to sum to 1

    return F.cross_entropy(output_logits, exp_encoding)


def l_norm(output_vector, target_angle, target_vector, l=2):
    return torch.norm(output_vector - target_vector, p=l, dim=1).mean()

def mse(output_angle, target_angle, target_vector):
    return F.mse_loss(output_angle.squeeze(), torch.deg2rad(target_angle))
