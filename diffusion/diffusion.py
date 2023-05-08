import matplotlib.pyplot as plt

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

if __name__ == "__main__":
    
    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8)
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = 128,
        timesteps = 1000,           # number of steps
        sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        loss_type = 'l1'            # L1 or L2
    )

    trainer = Trainer(
        diffusion,
        '../data/all-dogs-cropped',
        train_batch_size = 64,
        train_lr = 8e-5,
        train_num_steps = 12000,            # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True,                       # turn on mixed precision
        calculate_fid = False              # whether to calculate fid during training
    )

    trainer.train()