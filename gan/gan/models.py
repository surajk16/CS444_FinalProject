import torch


class Discriminator(torch.nn.Module):

    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
    
        self.conv1 = torch.nn.Conv2d(input_channels, 128, 4, 2, 1)
        self.activation = torch.nn.LeakyReLU(0.2)
        self.conv2 = torch.nn.Conv2d(128, 256, 4, 2, 1)
        self.conv3 = torch.nn.Conv2d(256, 512, 4, 2, 1)
        self.conv4 = torch.nn.Conv2d(512, 1024, 4, 2, 1)
        self.conv5 = torch.nn.Conv2d(1024, 1, 4, 1, 1) 
        self.batch_norm1 = torch.nn.BatchNorm2d(256)
        self.batch_norm2 = torch.nn.BatchNorm2d(512)
        self.batch_norm3 = torch.nn.BatchNorm2d(1024)
    
    def forward(self, x):

        x = self.activation(self.conv1(x))
        x = self.batch_norm1(self.activation(self.conv2(x)))
        x = self.batch_norm2(self.activation(self.conv3(x)))
        x = self.batch_norm3(self.activation(self.conv4(x)))
        x = self.activation(self.conv5(x))
        
        return x

class Generator(torch.nn.Module):
 
    
    def __init__(self, noise_dim, imsize=64, output_channels=3):
        super(Generator, self).__init__()
        
        self.noise_dim = noise_dim
        self.imsize = imsize
        
        self.conv1 = torch.nn.ConvTranspose2d(noise_dim, 1024, 4, 1, 0)
        self.conv2 = torch.nn.ConvTranspose2d(1024, 512, 4, 2, 1)
        self.conv3 = torch.nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.conv4 = torch.nn.ConvTranspose2d(256, 128, 4, 2, 1)
        
        if (imsize == 64):
            self.conv5 = torch.nn.ConvTranspose2d(128, 3, 4, 2, 1)
        else:
            self.conv5 = torch.nn.ConvTranspose2d(128, 64, 4, 2, 1)
            self.conv6 = torch.nn.ConvTranspose2d(64, 3, 4, 2, 1)
        self.batch_norm1 = torch.nn.BatchNorm2d(1024) 
        self.batch_norm2 = torch.nn.BatchNorm2d(512) 
        self.batch_norm3 = torch.nn.BatchNorm2d(256) 
        self.batch_norm4 = torch.nn.BatchNorm2d(128)
        self.batch_norm5 = torch.nn.BatchNorm2d(64)
        self.activationrel = torch.nn.ReLU()
        self.activationtanh = torch.nn.Tanh() 
    
    def forward(self, x):
        
        x = self.batch_norm1(self.activationrel(self.conv1(x)))
        x = self.batch_norm2(self.activationrel(self.conv2(x)))
        x = self.batch_norm3(self.activationrel(self.conv3(x)))
        x = self.batch_norm4(self.activationrel(self.conv4(x)))
        
        if (self.imsize == 64):
            x = self.activationtanh(self.conv5(x))
        else:
            x = self.batch_norm5(self.activationrel(self.conv5(x)))
            x = self.activationtanh(self.conv6(x))
        
        return x
    

