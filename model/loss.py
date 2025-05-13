import torch.nn.functional as F
import torch


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



#simultaneously generating nice distribution for non-anomalous vectors and pushing anomalous vectors away in the distribution
def full_loss(output, target, label, crop_ratio=0.1):
     
    z, mean, log_var = output 
    # dist_match_loss, mean, log_var, z = output

    r_loss = recreation_loss(z, target, label, crop_ratio)

    if torch.sum(~label) > 0:
        a_loss = 0#dist_match_loss
        # a_loss = anomaly_embedding_loss(x_normal, x_anomaly)
    else: 
        a_loss = 0

    loss = 5*r_loss + a_loss

    if torch.isnan(loss):
         print('oops')

    return loss


def anomaly_embedding_loss(x_normal, x_anomaly):


    x_norm_avg = x_normal.mean(dim=0)

    epsilon = 1e-3

    distances = torch.norm(x_anomaly-x_norm_avg, dim=-1)

    loss = -torch.log1p(distances + epsilon).mean() #maximizing distance of anomalous vectors. consider log1p

    return loss


def recreation_loss(output, target, label, crop_ratio):

    # MAKE SURE TO ONLY CONSIDER VECTORS FROM THE NORMAL DISTRIBUTION

    target = target[label]

    output_cropped, target_cropped = center_crop_match(output, target, crop_ratio)

    loss = F.mse_loss(output_cropped, target_cropped)

    return loss


