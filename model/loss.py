import torch.nn.functional as F


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
def Anomaly_VAE_loss(output, target, crop_ratio=0.8):


    output_cropped, target_cropped = center_crop_match(output, target)

    loss = F.mse_loss(output_cropped, target_cropped)

    return loss
