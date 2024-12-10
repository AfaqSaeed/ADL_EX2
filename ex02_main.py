import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
from ex02_model import Unet
from ex02_diffusion import Diffusion, linear_beta_schedule
from torchvision.utils import save_image

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network to diffuse images')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--timesteps', type=int, default=100, help='number of timesteps for diffusion model (default: 100)')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
    # parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    # parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=100, help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--run_name', type=str, default="DDPM")
    parser.add_argument('--dry_run', action='store_true', default=False, help='quickly check a single pass')
    return parser.parse_args()

def visualize_diffusion(images, diffusor, timesteps, store_path, reverse_transform):
    """
    Visualizes the forward diffusion process for a batch of images.

    Args:
        images (torch.Tensor): Batch of input images (original clean images).
        diffusor (Diffusion): The diffusion model instance.
        timesteps (list): Timesteps to visualize (e.g., [0, 10, 50, 100]).
        store_path (str): Directory path to save the visualization.
        reverse_transform (Compose): Transformation to convert back to human-readable format.
    """
    os.makedirs(store_path, exist_ok=True)
    images = images[:8]  # Use the first 8 images for visualization
    noise = torch.randn_like(images).to(images.device)

    visualizations = []
    for t in range(0,timesteps,10):
        t_tensor = torch.full((images.size(0),), t, device=images.device).long()
        print(f"Visualizing timesteps: {t}")
        noised_images = diffusor.q_sample(images, t_tensor, noise=noise, device="cpu")
        print(f"Shape of noised images: {noised_images.shape}")

        # Apply reverse_transform to convert back to human-readable format
        for img in noised_images:
            visualizations.append(reverse_transform(img.cpu()))

    # Save images step-by-step as a grid
    for idx, img in enumerate(visualizations):
        img.save(os.path.join(store_path, f"timestep_{idx // 8}_image_{idx % 8}.png"))
    print(f"Visualization saved to {store_path}")


def sample_and_save_images(n_images, diffusor, model, device, store_path,reverse_transform):
    # TODO: Implement - adapt code and method signature as needed
    model.eval()  # Set model to evaluation mode
    generated_images = diffusor.sample(
        model=model, 
        image_size=diffusor.img_size, 
        batch_size=n_images, 
        channels=3
    )
    os.makedirs(store_path, exist_ok=True)
    # fig = plt.figure(figsize=(10, 10))
    for i in range(len(generated_images)):
        
        plt.imshow(reverse_transform(generated_images[i].cpu()))
        
        plt.savefig(os.path.join(store_path, f"generated_image_{i}.png"))
        

    
    print(f"Generated images saved to {store_path}")


def test_without_vis(model, testloader, diffusor, device, args):
    # TODO: Implement - adapt code and method signature as needed
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    with torch.no_grad():
        for step, (images, labels) in enumerate(tqdm(testloader, desc="Testing")):
            images = images.to(device)
            t = torch.randint(0, args.timesteps, (len(images),), device=device).long()
            loss = diffusor.p_losses(model, images, t, loss_type="l2")
            total_loss += loss.item()

    print(f"Test Loss: {total_loss / len(testloader)}")

    

def train(model, trainloader, optimizer, diffusor, epoch, device, args):
    batch_size = args.batch_size
    timesteps = args.timesteps

    pbar = tqdm(trainloader)
    for step, (images, labels) in enumerate(pbar):

        images = images.to(device)
        optimizer.zero_grad()

        # Algorithm 1 line 3: sample t uniformly for every example in the batch
        t = torch.randint(0, timesteps, (len(images),), device=device).long()
        loss = diffusor.p_losses(model, images, t, loss_type="l2")

        loss.backward()
        optimizer.step()

        if step % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, step * len(images), len(trainloader.dataset),
                100. * step / len(trainloader), loss.item()))
        if args.dry_run:
            break


def test_vis(model, testloader, diffusor, device, args):
    # TODO (2.2): implement testing functionality, including generation of stored images.
    test_without_vis(model, testloader, diffusor, device, args)
    # Generate and save test images
    store_path = f"./results/{args.run_name}"
    sample_and_save_images(8, diffusor, model, device, store_path)



def run(args):
    timesteps = args.timesteps
    image_size = 32  # TODO (2.5): Adapt to new dataset
    channels = 3
    epochs = args.epochs
    batch_size = args.batch_size
    device = "cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu"

    model = Unet(dim=image_size, channels=channels, dim_mults=(1, 2, 4,)).to(device)
    model.load_state_dict(torch.load(os.path.join(r"C:\Study\Advanced Deep Learning\Exercises\Exercise 2\models", args.run_name, f"ckpt.pt"),weights_only=True))
    optimizer = AdamW(model.parameters(), lr=args.lr)

    my_scheduler = lambda x: linear_beta_schedule(0.0001, 0.02, x)
    diffusor = Diffusion(timesteps, my_scheduler, image_size, device)

    # define image transformations (e.g. using torchvision)
    transform = Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),    # turn into torch Tensor of shape CHW, divide by 255
        transforms.Lambda(lambda t: (t * 2) - 1)   # scale data to [-1, 1] to aid diffusion process
    ])
    reverse_transform = Compose([
        Lambda(lambda t: (t.clamp(-1, 1) + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        ToPILImage(),
    ])

    dataset = datasets.CIFAR10(r'C:\Study\Advanced Deep Learning\Exercises\Exercise 2\cifar10\train', download=True, train=True, transform=transform)
    trainset, valset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)])
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    # Download and load the test data
    testset = datasets.CIFAR10(r'C:\Study\Advanced Deep Learning\Exercises\Exercise 2\cifar10\test', download=True, train=False, transform=transform)
    testloader = DataLoader(testset, batch_size=int(batch_size/2), shuffle=True)

    # for epoch in range(epochs):
        # train(model, trainloader, optimizer, diffusor, epochs, device, args)
        # test_without_vis(model, valloader, diffusor, device, args)

    # test_vis(model, testloader, diffusor, device, args)

    save_path = r"C:\Study\Advanced Deep Learning\Exercises\Exercise 2\results"  # TODO: Adapt to your needs
    n_images = 8
    sample_and_save_images(n_images, diffusor, model, device, save_path,reverse_transform)
    torch.save(model.state_dict(), os.path.join(r"C:\Study\Advanced Deep Learning\Exercises\Exercise 2\models", args.run_name, f"ckpt.pt"))
    # Visualization
    for images, _ in trainloader:
        visualize_diffusion(images, diffusor, timesteps=timesteps, store_path=save_path,reverse_transform=reverse_transform)
        break


if __name__ == '__main__':
    args = parse_args()
    # TODO (2.2): Add visualization capabilities
    run(args)
