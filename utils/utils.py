from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import patchify
import numpy as np
import matplotlib.gridspec as gridspec
import glob as glob
import os
import cv2

def show_patches(patches):
    """
    The show_patches function accepts the image patches and displays them using Matplotlib if SHOW_PATCHES is True.
    """
    plt.figure(figsize=(patches.shape[0], patches.shape[1]))
    gs = gridspec.GridSpec(patches.shape[0], patches.shape[1])
    gs.update(wspace=0.01, hspace=0.02)
    counter = 0
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            ax = plt.subplot(gs[counter])
            plt.imshow(patches[i, j, 0, :, :, :])
            plt.axis('off')
            counter += 1
    plt.show()


def create_patches(
    input_paths, out_hr_path, out_lr_path, SHOW_PATCHES = False, STRIDE = 14, SIZE = 32
):
    """
    Make patches dataset from original image dataset.
    Patches are made into high resolution image and low resolution image.

    - Parameters
        input_paths: Original Images.
        out_hr_path: Saving path of high resolution patches.
        out_lr_path: Saving path of low resolution patches.
        SHOW_PATCHES: If True, Plot patches as examples.
        STRIDE: Make patches with strides.
        SIZE: Patch size.
    """
    os.makedirs(out_hr_path, exist_ok=True)
    os.makedirs(out_lr_path, exist_ok=True)
    all_paths = []
    for input_path in input_paths:
        all_paths.extend(glob.glob(f"{input_path}/*"))
    print(f"Creating patches for {len(all_paths)} images")
    for image_path in tqdm(all_paths, total=len(all_paths)):
        image = Image.open(image_path)
        image_name = image_path.split(os.path.sep)[-1].split('.')[0]
        w, h = image.size
        # Create patches of size (32, 32, 3)
        patches = patchify.patchify(np.array(image), (SIZE, SIZE, 3), STRIDE)
        if SHOW_PATCHES:
            show_patches(patches)
        counter = 0
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                counter += 1
                patch = patches[i, j, 0, :, :, :]
                patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    f"{out_hr_path}/{image_name}_{counter}.png",
                    patch
                )
                # Convert to bicubic and save.
                h, w, _ = patch.shape
                low_res_img = cv2.resize(patch, (int(w*0.5), int(h*0.5)), 
                                        interpolation=cv2.INTER_CUBIC)
                # Now upscale using BICUBIC.
                high_res_upscale = cv2.resize(low_res_img, (w, h), 
                                            interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(
                    f"{out_lr_path}/{image_name}_{counter}.png",
                    high_res_upscale
                )

def make_inference_images(paths, scale_factor_type):
    images = []
    for path in paths:
        images.extend(glob.glob(f"{path}/*.jpg"))
    print(len(images))
    # Select scaling-factor and set up directories according to that.
    if scale_factor_type == '2x':
        scale_factor = 0.5
        save_path_hr = path.replace('image', 'test_hr')
        os.makedirs(save_path_hr, exist_ok=True)
        save_path_lr = path.replace('image', 'test_bicubic_rgb_2x')
        os.makedirs(save_path_lr, exist_ok=True)
    if scale_factor_type == '3x':
        scale_factor = 0.333
        save_path_hr = path.replace('image', 'test_hr')
        os.makedirs(save_path_hr, exist_ok=True)
        save_path_lr = path.replace('image', 'test_bicubic_rgb_3x')
        os.makedirs(save_path_lr, exist_ok=True)
    if scale_factor_type == '4x':
        scale_factor = 0.25
        save_path_hr = path.replace('image', 'test_hr')
        os.makedirs(save_path_hr, exist_ok=True)
        save_path_lr = path.replace('image', 'test_bicubic_rgb_4x')
        os.makedirs(save_path_lr, exist_ok=True)
    print(f"Scaling factor: {scale_factor_type}")
    print(f"Low resolution images save path: {save_path_lr}")
    
    for image in images:
        orig_img = Image.open(image)
        image_name = image.split(os.path.sep)[-1]
        w, h = orig_img.size[:]
        print(f"Original image dimensions: {w}, {h}")
        orig_img.save(f"{save_path_hr}/{image_name}")
        low_res_img = orig_img.resize((int(w*scale_factor), int(h*scale_factor)), Image.BICUBIC)
        # Upscale using BICUBIC.
        high_res_upscale = low_res_img.resize((w, h), Image.BICUBIC)
        high_res_upscale.save(f"{save_path_lr}/{image_name}")

    return save_path_lr, save_path_hr