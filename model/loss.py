import torch.nn.functional as F
import torch

def nll_loss(output, target):
    return F.nll_loss(output, target)

def crop(t, h, w):
        start_h = (t.shape[2] - h) // 2
        start_w = (t.shape[3] - w) // 2
        return t[:, :, start_h:start_h + h, start_w:start_w + w]

def center_crop_match(output, target, crop_ratio=0.1):
    _, _, h_out, w_out = output.shape
    _, _, h_tgt, w_tgt = target.shape

    min_h = min(h_out, h_tgt)
    min_w = min(w_out, w_tgt)

    # Apply crop to total dimension, then divide across top/bottom
    total_crop_h = int(min_h * crop_ratio)
    total_crop_w = int(min_w * crop_ratio)

    final_h = max(min_h - total_crop_h, 1)
    final_w = max(min_w - total_crop_w, 1)

    return crop(output, final_h, final_w), crop(target, final_h, final_w)




#our custom loss
def Anomaly_VAE_loss(output, target, label, crop_ratio=0.8):
    output_cropped, target_cropped = center_crop_match(output, target)

    # Standard reconstruction loss
    mse = F.mse_loss(output_cropped, target_cropped, reduction='none')
    per_sample_loss = mse.view(mse.size(0), -1).mean(dim=1)  # mean over each sample

    # Custom logic:
    # - Normal samples (label == 0): minimize MSE
    # - Anomalous samples (label == 1): maximize MSE → equivalent to minimizing -MSE

    # Convert label shape if needed
    if label.dim() == 0 or label.size(0) != per_sample_loss.size(0):
        label = label.view(-1)

    # Binary mask
    normal_mask = (label == 1).float()
    anomaly_mask = (label == 0).float()
    
    normal_reward = per_sample_loss * normal_mask
    anomaly_penalty = -torch.log(per_sample_loss + 1e-6) * anomaly_mask

    loss = (normal_reward + anomaly_penalty).mean()

    return loss


# def Anomaly_VAE_loss(output, target, label, crop_ratio=0.8):


#     output_cropped, target_cropped = center_crop_match(output, target)

#     loss = F.mse_loss(output_cropped, target_cropped)

#     return loss
