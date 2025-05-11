import torch


# #sort of sloppily imported here, could probably be placed somewhere else so it also doesn't have to be implemented in loss function
# def crop(t, h, w):
#         start_h = (t.shape[2] - h) // 2
#         start_w = (t.shape[3] - w) // 2
#         return t[:, :, start_h:start_h + h, start_w:start_w + w]


# def center_crop_match(output, target, crop_ratio=0.1):
#     _, _, h_out, w_out = output.shape
#     _, _, h_tgt, w_tgt = target.shape

#     min_h = min(h_out, h_tgt)
#     min_w = min(w_out, w_tgt)

#     # Apply crop to total dimension, then divide across top/bottom
#     total_crop_h = int(min_h * crop_ratio)
#     total_crop_w = int(min_w * crop_ratio)

#     final_h = max(min_h - total_crop_h, 1)
#     final_w = max(min_w - total_crop_w, 1)

#     return crop(output, final_h, final_w), crop(target, final_h, final_w)

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
