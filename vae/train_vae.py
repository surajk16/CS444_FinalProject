import torch
from torchvision import datasets, transforms
from torch import optim
from torch.autograd import Variable

from PIL import Image
import numpy as np

import matplotlib.pyplot as plt

import os
import argparse
import time
import json

from vae import VAE

def get_data_loader(path, batch_size):

    random_transforms = [transforms.ColorJitter(), transforms.RandomRotation(degrees=20)]
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(random_transforms, p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_data = datasets.ImageFolder(path, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
    
    return train_loader

def train(model, train_loader, optimizer, epochs):
    
    mse_losses = []
    kld_losses = []

    for epoch in range(1, epochs+1):
        start_time = time.time()
        model.train()
        print('Starting epoch: {}'.format(epoch))

        mse_loss_total = 0
        kld_loss_total = 0

        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            recon_batch, mu, logvar = model(data)
            mse_loss, kld_loss = model.loss_function(recon_batch, data, mu, logvar)

            mse_loss_total += mse_loss
            kld_loss_total += kld_loss

            loss = mse_loss + kld_loss

            loss.backward()
            optimizer.step()
            
        mse_losses.append(mse_loss_total)
        kld_losses.append(kld_loss_total)

        print("Loss: Total: {}, MSE: {}, KLD: {}".format(mse_loss_total+kld_loss_total, mse_loss_total, kld_loss_total))
        print('Time for epoch {}: {}'.format(epoch, time.time()-start_time))
        
    return mse_losses, kld_losses

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cropped', action='store_true')
    parser.add_argument('--non-cropped', dest='cropped', action='store_false')
    parser.set_defaults(cropped=True)
    
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--latent_dim", type=int, required=True)
    parser.add_argument("--num_images", type=int, required=True)
    args = parser.parse_args()
    
    if (args.cropped):
        data_path = "/home/sk118/CS444_FinalProject/data/all-dogs-cropped/"
        expt_name = "cropped"
    else:
        data_path = "/home/sk118/CS444_FinalProject/data/all-dogs/"
        expt_name = "not_cropped"
    
    data_loader = get_data_loader(data_path, args.batch_size)

    model = VAE(latent_dim=args.latent_dim, batch_size=args.batch_size).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    mse_losses, kld_losses = train(model, data_loader, optimizer, args.epochs)
    
    mse_losses = torch.tensor(mse_losses, device = 'cpu').numpy()
    kld_losses = torch.tensor(kld_losses, device = 'cpu').numpy()
    
    result_dir = "/home/sk118/CS444_FinalProject/vae/results/{}_k_{}_epoch_{}_batch_size_{}_lr_{}/".format(expt_name, args.latent_dim, args.epochs, args.batch_size, args.lr)
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        
    if not os.path.exists(result_dir + "generated/"):
        os.makedirs(result_dir + "generated/")
    
    plt.plot(list(range(1, len(mse_losses)+1)), mse_losses)
    plt.title('MSE loss curve for non-cropped')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.savefig(result_dir + 'mse_loss_curve.png')
    plt.close()
    
    plt.plot(list(range(1, len(kld_losses)+1)), kld_losses)
    plt.title('KLD loss curve for non-cropped')
    plt.xlabel('Epochs')
    plt.ylabel('KLD Loss')
    plt.savefig(result_dir + 'kld_loss_curve.png')
    plt.close()
    
    samples = Variable(torch.randn(args.num_images, args.latent_dim, 4, 4)).to(device)
    samples = model.decoder(samples).detach().cpu().numpy().transpose(0, 2, 3, 1)

    for i, img in enumerate(samples):
        
        im = Image.fromarray((img * 255).astype(np.uint8))
        im.save(result_dir + "generated/{}.png".format(i))
    
    