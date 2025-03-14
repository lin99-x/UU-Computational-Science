import math
import sys
import io
from typing import Iterable

import torch
import torch.nn as nn
from torchvision.transforms import ToPILImage

import util.misc as misc
import util.lr_sched as lr_sched

from visualizer import get_local
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable,      # provides batches of training data
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,                # a utility for scaling the loss
                    log_writer = None,
                    args = None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    
    accum_iter = args.accum_iter
    
    optimizer.zero_grad()                       # the optimizer's gradients are zeroed
    
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
        
    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler, adjust learning rate after each batch of data is processed
        # if data_iter_step % accum_iter == 0:
        #     lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():         # the forward pass is done in mixed precision, save memory
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter                      # scales the loss by the number of gradient accumulation steps
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)    # not using DDP, so this is a no-op
        
        # # get the attention maps of all layers
        # cache = get_local.cache
        # attention_maps = cache['Attention.forward']
        # print(len(attention_maps))

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            
            # # visualize the last layer of the attention map
            # visualize_grid_to_grid(attention_maps[-1][0, :, 0, 1:], samples[0])
            # # visualize_grid_to_grid_all_layers(attention_maps, args.mask_ratio, samples[0])
            # log_writer.add_figure('layer12', plt.gcf())
            
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}




# visualization functions
  
def visualize_head(att_map):
    ax = plt.gca()
    # Plot the heatmap
    im = ax.imshow(att_map)
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
def visualize_grid_to_grid(att_map, image, grid_size=14, alpha=0.6):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
    
    # inpout att_map shape: (N, num_heads, H, W)

    image = image.permute(1, 2, 0).cpu().numpy()

    mask = np.mean(att_map, 0)

    mask = mask.reshape(grid_size[0], grid_size[1])

    mask = Image.fromarray(mask).resize((image.shape[1], image.shape[0]))

    
    fig, ax = plt.subplots(1, 2, figsize=(10,7))
    fig.tight_layout()
    
    ax[0].imshow(image, cmap='gray')
    ax[0].axis('off')
    
    ax[1].imshow(image, cmap='gray')
    ax[1].imshow(mask/np.max(mask), alpha = alpha, cmap='rainbow')
    ax[1].axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
def visualize_grid_to_grid_all_layers(att_map: list, mask_ratio, image, grid_size=14, alpha=0.6):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
        
    image = image.permute(1, 2, 0).cpu().numpy()
    print("image has shape: ", image.shape)
    
    for i in range(len(att_map)):
        layer = att_map[i]
        num_heads = layer.shape[1]
        attentions = layer[0, :, 0, 1:].reshape(num_heads, -1)
        mask = np.mean(attentions, 0)
        mask = Image.fromarray(mask).resize((image.shape[1], image.shape[0]))
        
        fig, ax = plt.subplots(1, 2, figsize=(10,7))
        fig.tight_layout()
        
        ax[0].imshow(image, cmap='gray')
        ax[0].axis('off')
        
        ax[1].imshow(image, cmap='gray')
        ax[1].imshow(mask/np.max(mask), alpha = alpha, cmap='rainbow')
        ax[1].axis('off')
        
        plt.title(f"Layer {i+1}")
        
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    

def plot_attention(img, attention):
    n_heads = attention.shape[0]

    plt.figure(figsize=(10, 10))
    text = ["Original Image", "Head Mean"]
    for i, fig in enumerate([img, np.mean(attention, 0)]):
        plt.subplot(1, 2, i+1)
        plt.imshow(fig, cmap="inferno")
        plt.title(text[i])
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # plt.figure(figsize=(10, 10))
    # for i in range(n_heads):
    #     plt.subplot(n_heads//4, 4, i+1)
    #     plt.imshow(attention[i], cmap='inferno')
    #     plt.title(f"Head n: {i+1}")
    # plt.tight_layout()
    # plt.savefig(buf, format='png')
    # buf.seek(0)