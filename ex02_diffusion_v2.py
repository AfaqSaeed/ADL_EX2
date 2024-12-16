import torch
import torch.nn.functional as F
from ex02_helpers import *
from tqdm import tqdm


def linear_beta_schedule(beta_start, beta_end, timesteps):
    """
    standard linear beta/variance schedule as proposed in the original paper
    """
    return torch.linspace(beta_start, beta_end, timesteps)


# TODO: Transform into task for students
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    also read:
    https://www.analyticsvidhya.com/blog/2024/07/noise-schedules-in-stable-diffusion/
    """
    # TODO (2.3): Implement cosine beta/variance schedule as discussed in the paper mentioned above
    pass


def sigmoid_beta_schedule(beta_start, beta_end, timesteps):
    """
    sigmoidal beta schedule - following a sigmoid function
    """
    # TODO (2.3): Implement a sigmoidal beta schedule. Note: identify suitable limits of where you want to sample the sigmoid function.
    # Note that it saturates fairly fast for values -x << 0 << +x

    pass


class Diffusion:

    # TODO (2.4): Adapt all methods in this class for the conditional case. You can use y=None to encode that you want to train the model fully unconditionally.

    def __init__(self, timesteps, get_noise_schedule, img_size, device="cuda"):
        """
        Takes the number of noising steps, a function for generating a noise schedule as well as the image size as input.
        """
        self.timesteps = timesteps

        self.img_size = img_size
        self.device = device

        # define beta schedule
        self.betas = get_noise_schedule(self.timesteps).to(self.device)

        # TODO (2.2): Compute the central values for the equation in the forward pass already here so you can quickly use them in the forward pass.
        # Note that the function torch.cumprod may be of help

        # define alphas
        
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas,dim=0).to(self.device)
        self.alpha_bar_cumprod_prev = torch.cat(
                    [torch.tensor([1.0], device=self.device), self.alpha_bar[:-1]])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1-self.alpha_bar)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alpha_bar_cumprod_prev) / (1.0 - self.alpha_bar)  # equivalent to At/As

        

        

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        # TODO (2.2): implement the reverse diffusion process of the model for (noisy) samples x and timesteps t. Note that x and t both have a batch dimension

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        predicted_noise = model(x, t)

        mean = 1 / torch.sqrt(self.alphas[t_index]) * (x- (self.betas[t_index] * predicted_noise) / (self.sqrt_one_minus_alpha_bar[t_index]) )

        # TODO (2.2): The method should return the image at timestep t-1.
        if t_index > 0:
            noise = torch.randn_like(x)
            variance = torch.sqrt(self.posterior_variance[t_index])
            sample = mean + variance * noise
        else:
            sample = mean
        return sample

    # Algorithm 2 (including returning all images)
    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=3):
        # TODO (2.2): Implement the full reverse diffusion loop from random noise to an image, iteratively ''reducing'' the noise in the generated image.

        # TODO (2.2): Return the generated images
        x = torch.randn((batch_size, channels, image_size, image_size), device=self.device)
        for rvs_timestep in tqdm(reversed(range(self.timesteps)), desc="Sampling"):
            t = torch.full((batch_size,), rvs_timestep, device=self.device, dtype=torch.long)
            x = self.p_sample(model, x, t, rvs_timestep)
        return x
    
    # forward diffusion (using the nice property)
    def q_sample(self, x_zero, t, noise=None):
        # TODO (2.2): Implement the forward diffusion process using the beta-schedule defined in the constructor; if noise is None, you will need to create a new noise vector, otherwise use the provided one.
        if noise==None:
            noise = torch.randn_like(x_zero)

        sqrt_alpha_bar = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
        return  sqrt_alpha_bar*x_zero + sqrt_one_minus_alpha_bar*noise
        

    def p_losses(self, denoise_model, x_zero, t, noise=None, loss_type="l1"):
        # TODO (2.2): compute the input to the network using the forward diffusion process and predict the noise using the model; if noise is None, you will need to create a new noise vector, otherwise use the provided one.

        if noise is None:
            noise = torch.randn_like(x_zero)
        
        x_t = self.q_sample(x_zero, t, noise)

        # Predict noise using the model
        predicted_noise = denoise_model(x_t, t)

        if loss_type == 'l1':
            # TODO (2.2): implement an L1 loss for this task
            loss = F.l1_loss(predicted_noise, noise)
        elif loss_type == 'l2':
            # TODO (2.2): implement an L2 loss for this task
            loss = F.mse_loss(predicted_noise,noise)
        else:
            raise NotImplementedError()

        return loss
