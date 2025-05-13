import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from torchvision import transforms
import matplotlib.pyplot as plt
import time

import os
from torchvision.utils import save_image

# Set output directory
os.makedirs("output_images", exist_ok=True)

def save_side_by_side(input_tensor, output_tensor, label):
    input_img = unnormalize(input_tensor)
    output_img = unnormalize(output_tensor)

    # Combine input and output side-by-side
    combined = torch.cat((input_img, output_img), dim=2)  # concatenate along width

    filename = f"output_images/_label_{label}.png"
    save_image(combined, filename)
    print(f"Saved: {filename}")

# Inverse normalization
inv_normalize = transforms.Normalize(
    mean=[-m/s for m, s in zip((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))],
    std=[1/s for s in (0.5, 0.5, 0.5)]
)

def unnormalize(tensor):
    return inv_normalize(tensor.cpu()).clamp(0, 1)

def show_side_by_side(input_tensor, output_tensor):
    input_img = transforms.ToPILImage()(unnormalize(input_tensor))
    output_img = transforms.ToPILImage()(unnormalize(output_tensor))

    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    axs[0].imshow(input_img)
    axs[0].set_title("Input")
    axs[0].axis("off")
    axs[1].imshow(output_img)
    axs[1].set_title("Reconstruction")
    axs[1].axis("off")
    plt.show()


def main(config):
    logger = config.get_logger('test')


    #Cross compatability windows to linux
    import pathlib
    import sys

    # Cross-platform patch to handle incompatible pathlib types in PyTorch checkpoints
    if hasattr(pathlib, "WindowsPath") and sys.platform != "win32":
        pathlib.WindowsPath = pathlib.PosixPath
    elif hasattr(pathlib, "PosixPath") and sys.platform == "win32":
        pathlib.PosixPath = pathlib.WindowsPath

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=5,
        shuffle=True,
        validation_split=0.0,
        training=False,
        num_workers=2
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    # logger.info(model)

    # get function handles of loss and metrics
    # loss_fn = getattr(module_loss, config['loss'])
    # metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # logger.info('Loading checkpoint: {} ...'.format(config.resume))
    # checkpoint = torch.load(config.resume, weights_only=False)
    # state_dict = checkpoint['state_dict']
    # if config['n_gpu'] > 1:
    #     model = torch.nn.DataParallel(model)
    # model.load_state_dict(state_dict)

    # # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = model.to(device)
    # model.eval()

    # total_loss = 0.0
    # total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for _, (data, label) in enumerate(tqdm(data_loader)):
            data = data.to(device)

            # fake_label = torch.ones_like(label) # pretending all samples are normal for testing / output
            # output = model(data, fake_label)
            output = model(data, label)


            result = output[0]

            target=data

            for i in range(data.shape[0]):
                result_i = result[i,:]
                target_i = target[i,:]
                print(f'image/result {str(i+1)}, label: {label[i]} shown')
                show_side_by_side(target_i, result_i)
                save_side_by_side(target_i, result_i, label=str(time.time())[-6:-1])
                


            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            # loss = loss_fn(output, target, label)
            # batch_size = data.shape[0]
            # total_loss += loss.item() * batch_size
            # for i, metric in enumerate(metric_fns):
            #     total_metrics[i] += metric(output, target) * batch_size

    # n_samples = len(data_loader.sampler)
    # log = {'loss': total_loss / n_samples}
    # log.update({
    #     met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    # })
    # logger.info(log)
    pass


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
