from __future__ import print_function
import argparse
import ast
import random  # to set the python random seed
import os
from math import ceil

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from IPython.display import HTML
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.utils as vutils
import torch.optim as optim
from PIL import Image
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torchvision.datasets as dset
# Ignore excessive warnings
import logging

from torchvision.transforms import ToTensor

logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)

# Set random seed for reproducibility
manualSeed = 42
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Parameters

log_interval = 10

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 256

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 1024

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 300

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

scale = 0.1

num_predict_image = 32

no_cuda = False

dataroot = 'data/images'
gene_expression_data = 'data/gene_expression/data_1024.csv'

random_indices = random.sample(range(13000), num_predict_image)

losses_d = []
losses_g = []


# Model Definition

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),
            # state size. (ngf // 2) x 128 x 128
            nn.ConvTranspose2d(ngf // 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 256 x 256
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 256 x 256
            nn.Conv2d(nc, ndf // 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf // 2) x 128 x 128
            nn.Conv2d(ndf // 2, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 4 x 4
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# Defining the Training Function

def load_gene_expression(data_path='data.csv'):
    df = pd.read_csv(data_path)
    gene_expression = df['gene_expression']
    image_path = df['image_path'].tolist()
    new_gene_expression = []
    new_image_path = []
    for i in range(len(gene_expression)):
        try:
            row = ast.literal_eval(gene_expression[i])
            new_gene_expression.append(row)
            new_image_path.append(image_path[i])
        except Exception as e:
            print('error : ', e)
    # gene_expression = [ast.literal_eval(row) for row in gene_expression]
    gene_expression = np.array(new_gene_expression)
    print('max: ', np.max(gene_expression), 'min: ', np.min(gene_expression))
    return gene_expression, image_path


# scale matrix to [-1, 1]
def scale_matrix(matrix, max):
    scaled_matrix = 2 * matrix / max - 1
    return scaled_matrix


def plot_images(images, filename):
    h, w, c = images.shape[1:]
    grid_size = ceil(np.sqrt(images.shape[0]))
    images = (images + 1) / 2. * 255.
    images = images.astype(np.uint8)
    images = (images.reshape(grid_size, grid_size, h, w, c)
              .transpose(0, 2, 1, 3, 4)
              .reshape(grid_size * h, grid_size * w, c))
    # plt.figure(figsize=(16, 16))
    plt.imsave(filename, images)
    plt.imshow(images)
    plt.show()


def plot_losses(losses_d, losses_g, filename):
    fig, axes = plt.subplots(1, 2, figsize=(8, 2))
    axes[0].plot(losses_d)
    axes[1].plot(losses_g)
    axes[0].set_title("losses_d")
    axes[1].set_title("losses_g")
    plt.tight_layout()
    plt.savefig(filename)
    # plt.close()


def load_image():
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)
    return dataloader


def train(gen, disc, device, dataloader, optimizerG, optimizerD, criterion, epoch, iters):
    gen.train()
    disc.train()
    # fixed_noise = torch.randn(64, config.nz, 1, 1, device=device)

    # Load gene expression data
    gene_expression, image_paths = load_gene_expression(gene_expression_data)

    # process real image
    image_fixed = [image_paths[i] for i in random_indices]

    image_tensors = []
    for img_path in image_fixed:
        img = Image.open(img_path)
        img_tensor = ToTensor()(img)  # Convert to tensor
        image_tensors.append(img_tensor)

    # Convert the list of tensors to a single tensor
    real_images = torch.stack(image_tensors)

    gene_expression_fixed = np.array([gene_expression[i] for i in random_indices])
    tensor = torch.tensor(gene_expression_fixed, dtype=torch.float32)
    tensor = tensor.unsqueeze(2).unsqueeze(3)
    fixed_noise = tensor.to(device)

    random_noise = torch.randn(num_predict_image, nz, 1, 1, device=device)
    # Min and max values in the noise tensor
    min_noise, max_noise = random_noise.min(), random_noise.max()

    # Scale the noise to [-1, 1]
    scaled_noise = 2 * (random_noise - min_noise) / (max_noise - min_noise) - 1
    scaled_noise = scaled_noise.to(device)
    fixed_noise = fixed_noise + scale * scaled_noise

    # print('fixed_noise: ', fixed_noise)
    # Establish convention for real and fake labels during training (with label smoothing)
    real_label = 0.9
    fake_label = 0.1
    for i, data in enumerate(dataloader, 0):
        # *****
        # Update Discriminator
        # *****
        ## Train with all-real batch
        disc.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)
        # Forward pass real batch through D
        output = disc(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        gene_expression_batch = gene_expression[i * b_size:(i + 1) * b_size]

        # noise = torch.randn(b_size, config.nz, 1, 1, device=device)
        # Generate fake image batch with G
        tensor_gene = torch.tensor(gene_expression_batch, dtype=torch.float32)
        tensor_gene = tensor_gene.unsqueeze(2).unsqueeze(3)
        noise = tensor_gene.to(device)

        random_noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Min and max values in the noise tensor
        min_noise, max_noise = random_noise.min(), random_noise.max()

        # Scale the noise to [-1, 1]
        scaled_noise = 2 * (random_noise - min_noise) / (max_noise - min_noise) - 1
        scaled_noise = scaled_noise.to(device)
        noise = noise + scale * scaled_noise

        fake = gen(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = disc(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        # *****
        # Update Generator
        # *****
        gen.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = disc(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            losses_d.append(errD.item())
            losses_g.append(errG.item())
            if not os.path.exists('fake_image_1024'):
                os.makedirs('fake_image_1024')
            plot_losses(losses_d, losses_g, 'fake_image_1024/losses.png')

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                fake = gen(fixed_noise).detach().cpu()
                # print(fake.shape)
                if not os.path.exists('fake_image_1024'):
                    os.makedirs('fake_image_1024')

            stacked_images = torch.stack((real_images, fake), dim=1)

            # Reshape the tensor to mix the images
            mixed_images = stacked_images.view(-1, *real_images.shape[1:])

            # Generate grid of images
            grid = vutils.make_grid(mixed_images, padding=2, normalize=True, scale_each=True)

            # Save to file
            save_image(grid, f'fake_image_1024/generated_epoch_{epoch}_{i}.png')

        iters += 1


def main():
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Set random seeds and deterministic pytorch for reproducibility
    random.seed(manualSeed)  # python random seed
    torch.manual_seed(manualSeed)  # pytorch random seed
    np.random.seed(manualSeed)  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # Load the dataset
    trainloader = load_image()

    # Create the generator
    netG = Generator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(weights_init)

    # Create the Discriminator
    netD = Discriminator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    iters = 0
    for epoch in range(1, num_epochs + 1):
        train(netG, netD, device, trainloader, optimizerG, optimizerD, criterion, epoch, iters)
        if epoch % 50 == 0:
            torch.save(netG.state_dict(), f"model_epoch_{epoch}.h5")


if __name__ == '__main__':
    main()
    data_loss = []
    for loss_d, loss_g in zip(losses_d, losses_g):
        data_loss.append((loss_d, loss_g))
    data_loss = pd.DataFrame(data_loss, columns=['loss_d', 'loss_g'])
    data_loss.to_csv('fake_image_1024/losses.csv')
