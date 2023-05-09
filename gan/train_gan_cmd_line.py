import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

from gan.train import train
from gan.losses import discriminator_loss, generator_loss
from gan.models import Discriminator, Generator
from gan.utils import sample_noise, deprocess_img

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "gpu")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cropped', action='store_true')
    parser.add_argument('--non-cropped', dest='cropped', action='store_false')
    parser.set_defaults(cropped=True)
    
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--imsize", type=int, default=64)
    parser.add_argument("--latent_dim", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--num_images", type=int, required=True)
    
    parser.add_argument("--dog_class", type=str)
    
    args = parser.parse_args()
    
    if (args.cropped):
        data_path = "/home/sk118/CS444_FinalProject/data/all-dogs-cropped/"
        expt_name = "cropped"
    else:
        data_path = "/home/sk118/CS444_FinalProject/data/all-dogs/"
        expt_name = "not_cropped"

    dog_train = ImageFolder(root=data_path, transform=transforms.Compose([
        transforms.Resize(int(1.15 * args.imsize)),
        transforms.RandomCrop(args.imsize),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter()], p=0.2),
        transforms.ToTensor()
    ]))
    
    if (args.dog_class is not None):
        expt_name += "_{}".format(args.dog_class)
        class_idx = dog_train.class_to_idx[args.dog_class]
        indices = []
        for i, image in enumerate(dog_train):
            if(image[1] == class_idx):
                indices.append(i)
                
        dog_train = Subset(dog_train, indices)

    dog_loader_train = DataLoader(dog_train, batch_size=args.batch_size, drop_last=True, shuffle=True)
    
    D = Discriminator().to(device)
    G = Generator(noise_dim=args.latent_dim, imsize=args.imsize).to(device)
    
    D_optimizer = torch.optim.Adam(D.parameters(), lr=args.lr, betas = (0.5, 0.999))
    G_optimizer = torch.optim.Adam(G.parameters(), lr=args.lr, betas = (0.5, 0.999))
    
    train(
        D, 
        G, 
        D_optimizer, 
        G_optimizer, 
        discriminator_loss, 
        generator_loss, 
        num_epochs = args.epochs, 
        show_every = 10,
        batch_size = args.batch_size, 
        train_loader = dog_loader_train, 
        device = device,
        noise_size = args.latent_dim
    )
    
    result_dir = "/home/sk118/CS444_FinalProject/gan/results/{}_k_{}_epoch_{}_batch_size_{}_lr_{}/".format(expt_name, args.latent_dim, args.epochs, args.batch_size, args.lr)
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        
    if not os.path.exists(result_dir + "generated/"):
        os.makedirs(result_dir + "generated/")
    
    for i in range(args.num_images):
        noise = sample_noise(1, args.latent_dim).reshape(1, args.latent_dim, 1, 1).to(device)
        fake_image = G(noise).reshape(1, 3, args.imsize, args.imsize)
        disp_fake_image = deprocess_img(fake_image.data)
        img = (disp_fake_image).cpu().numpy()
        img = np.squeeze(img, axis=0)
        img = np.transpose(img, (1, 2, 0))
        
        im = Image.fromarray((img * 255).astype(np.uint8))
        im.save(result_dir + "generated/{}.png".format(i))
        
        
        
    
    

    
    