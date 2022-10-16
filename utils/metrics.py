import math
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image


def psnr(label, outputs, max_val=1.):
    """
    Compute Peak Signal to Noise Ratio (the higher the better).
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE).
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio#Definition
    Note that the output and label pixels (when dealing with images) should
    be normalized as the `max_val` here is 1 and not 255.
    """
    label = label.cpu().detach().numpy()
    outputs = outputs.cpu().detach().numpy()
    diff = outputs - label
    rmse = math.sqrt(np.mean((diff) ** 2))
    if rmse == 0:
        return 100
    else:
        PSNR = 20 * math.log10(max_val / rmse)
        return PSNR


def save_plot(train_loss, val_loss, train_psnr, val_psnr, result_path):
    """
    The functions to save the loss and PSNR graphs for training and validation.
    """
    # Loss plots.
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(val_loss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(result_path, 'loss.png'))
    plt.close()
    # PSNR plots.
    plt.figure(figsize=(10, 7))
    plt.plot(train_psnr, color='green', label='train PSNR dB')
    plt.plot(val_psnr, color='blue', label='validataion PSNR dB')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.savefig(os.path.join(result_path, 'psnr.png'))
    plt.close()


def save_model(epochs, model, optimizer, criterion, result_path):
    """
    Function to save the trained model to disk.
    """
    # Remove the last model checkpoint if present.
    save_path = os.path.join(result_path, 'model_ckpt.pth')
    torch.save({
                'epoch': epochs+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, save_path)

def save_validation_results(outputs, epoch, batch_iter, result_path):
    """
    Function to save the validation reconstructed images.
    """
    file_name = "val_sr_" + str(epoch) + "_" + str(batch_iter) + ".png"
    save_path = os.path.join(result_path, file_name)
    save_image(
        outputs, 
        save_path
    )