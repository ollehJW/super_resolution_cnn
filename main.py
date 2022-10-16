## 0. Import Packages
import argparse
import os
from utils import psnr, save_plot, save_model, save_validation_results, create_patches, make_inference_images, get_dataloaders
from model import SRCNN
import torch
import time
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

SAVE_VALIDATION_RESULTS = True

def train(model, dataloader):
    model.train()
    running_loss = 0.0
    running_psnr = 0.0
    for bi, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        image_data = data[0].to(device)
        label = data[1].to(device)
        
        # Zero grad the optimizer.
        optimizer.zero_grad()
        outputs = model(image_data)
        loss = criterion(outputs, label)
        # Backpropagation.
        loss.backward()
        # Update the parameters.
        optimizer.step()
        # Add loss of each item (total items in a batch = batch size).
        running_loss += loss.item()
        # Calculate batch psnr (once every `batch_size` iterations).
        batch_psnr =  psnr(label, outputs)
        running_psnr += batch_psnr
    final_loss = running_loss/len(dataloader.dataset)
    final_psnr = running_psnr/len(dataloader)
    return final_loss, final_psnr

    
def validate(model, dataloader, epoch, path):
    model.eval()
    running_loss = 0.0
    running_psnr = 0.0
    with torch.no_grad():
        for bi, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            image_data = data[0].to(device)
            label = data[1].to(device)
            
            outputs = model(image_data)
            loss = criterion(outputs, label)
            # Add loss of each item (total items in a batch = batch size) .
            running_loss += loss.item()
            # Calculate batch psnr (once every `batch_size` iterations).
            batch_psnr = psnr(label, outputs)
            running_psnr += batch_psnr
            # For saving the batch samples for the validation results
            # every 500 epochs.
            if SAVE_VALIDATION_RESULTS and (epoch % 5) == 0:
                save_validation_results(outputs, epoch, bi, path)
    final_loss = running_loss/len(dataloader.dataset)
    final_psnr = running_psnr/len(dataloader)
    return final_loss, final_psnr




if __name__ == '__main__':

    ## 1. Arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_image_path', default = './data/train/image', type=str, required=True)
    parser.add_argument('--test_image_path', default = './data/test/image', type=str, required=True)
    parser.add_argument('--result_dir', default = './result', type=str, required=True)
    parser.add_argument('--scale_factor', type=str, default='2x')
    parser.add_argument('--lr', default = 0.001, type=float)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=400)
    args = parser.parse_args()

    ## 2. Make Patches from Image dataset
    hr_path = args.train_image_path.replace('image', 'hr_patches')
    lr_path = args.train_image_path.replace('image', 'lr_patches')
    create_patches([args.train_image_path], hr_path, lr_path, SHOW_PATCHES=False, STRIDE=14, SIZE=32)
    
    ## 3. Make Test Dataset
    test_input_path, test_label_path = make_inference_images([args.test_image_path], scale_factor_type=args.scale_factor)

    ## 4. Construct Torch dataloader
    train_loader, valid_loader = get_dataloaders(train_image_paths=lr_path, train_label_paths=hr_path, valid_image_path=test_input_path, valid_label_paths=test_label_path, TRAIN_BATCH_SIZE=args.batch_size, TEST_BATCH_SIZE=1)

    ## 5. 
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(os.path.join(args.result_dir, 'valid_results'), exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Computation device: ', device)

    # 6. Build Model
    model = SRCNN().to(device)
    print(model)

    ## 7. Optimizer, Loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    ## 8. Train
    train_loss, val_loss = [], []
    train_psnr, val_psnr = [], []
    start = time.time()
    for epoch in range(args.epoch):
        print(f"Epoch {epoch + 1} of {args.epoch}")
        train_epoch_loss, train_epoch_psnr = train(model, train_loader)
        val_epoch_loss, val_epoch_psnr = validate(model, valid_loader, epoch+1, os.path.join(args.result_dir, 'valid_results'))
        print(f"Train PSNR: {train_epoch_psnr:.3f}")
        print(f"Val PSNR: {val_epoch_psnr:.3f}")
        train_loss.append(train_epoch_loss)
        train_psnr.append(train_epoch_psnr)
        val_loss.append(val_epoch_loss)
        val_psnr.append(val_epoch_psnr)
    
        # Save model with all information every 100 epochs. Can be used 
        # resuming training.
        if (epoch+1) % 100 == 0:
            save_model(epoch, model, optimizer, criterion, args.result_dir)
        # Save the PSNR and loss plots every epoch.
        save_plot(train_loss, val_loss, train_psnr, val_psnr, args.result_dir)
    end = time.time()
    print(f"Finished training in: {((end-start)/60):.3f} minutes") 

