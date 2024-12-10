import torch
import torch.nn.functional as F
from ex02_helpers import extract
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
        self.betas = get_noise_schedule(self.timesteps)
        self.betas = self.betas.to(device)
        # TODO (2.2): Compute the central values for the equation in the forward pass already here so you can quickly use them in the forward pass.
        # Note that the function torch.cumprod may be of help

        # define alphas
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(device)  # Cumulative product


        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_bar = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1 - self.alpha_bars)
        self.one_over_sqrt_alphas = 1 / torch.sqrt(self.alphas)
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = torch.sqrt(1 - self.alpha_bars)

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        # TODO (2.2): implement the reverse diffusion process of the model for (noisy) samples x and timesteps t. Note that x and t both have a batch dimension
        # Predict noise using the model
        predicted_noise = model(x, t)
            # Equation 11 in the paper
    
        # Compute the mean (Equation 11 in the paper)
        beta_t = self.betas[t_index].view(-1, 1, 1, 1)
        sqrt_alpha_bar_t = self.sqrt_alphas_bar[t_index].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_bar[t_index].view(-1, 1, 1, 1)
        # print("predicted_noise",predicted_noise.is_cuda)
        # print("beta_t",beta_t.is_cuda)
        # print("sqrt_alpha_bar_t",sqrt_alpha_bar_t.is_cuda)
        # print("sqrt_one_minus_alpha_bar_t",sqrt_one_minus_alpha_bar_t.is_cuda)
        
        

    
        # Use our model (noise predictor) to predict the mean
        mean = (1 / sqrt_alpha_bar_t) * (x - (beta_t * predicted_noise / sqrt_one_minus_alpha_bar_t ))

        # Add noise if not the last timestep
        if t_index > 0:
            noise = torch.randn_like(x)
            posterior_variance = self.posterior_variance[t_index].view(-1, 1, 1, 1)
            x_prev = mean + torch.sqrt(posterior_variance) * noise
        else:
            x_prev = mean

        return x_prev

        # TODO (2.2): The method should return the image at timestep t-1.

    # Algorithm 2 (including returning all images)
    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=3):
        # TODO (2.2): Implement the full reverse diffusion loop from random noise to an image, iteratively ''reducing'' the noise in the generated image.
        # Note that you will need to generate noise for the first timestep and then iteratively reduce it using the model.
        # You will need to use the p_sample method for this task.
        # Initialize with pure Gaussian noise
        x = torch.randn((batch_size, channels, image_size, image_size), device=self.device)

        # Reverse diffusion loop
        for t_index in reversed(range(self.timesteps)):
            t_tensor = torch.full((batch_size,), t_index, device=self.device, dtype=torch.long)
            x = self.p_sample(model, x, t_tensor, t_index)

        return x
        # TODO (2.2): Return the generated images
        

    # forward diffusion (using the nice property)
    def q_sample(self, x_zero, t, noise=None,device="cuda"):
        # TODO (2.2): Implement the forward diffusion process using the beta-schedule defined in the constructor; if noise is None, you will need to create a new noise vector, otherwise use the provided one.
        if noise is None:
            noise = torch.randn_like(x_zero)
        noise = noise.to(device)
        x_zero = x_zero.to(device)

        # print("sqrt_alpha_bar_shape",self.sqrt_alphas_bar.shape)
        # print("sqrt_one_minus_alpha_bar_shape",self.sqrt_one_minus_alphas_bar.shape)
        # print("noise_shape",noise.shape)
        print("x_zero_shape",x_zero.shape)
        # print("t",t)
        sqrt_alpha_bar_t = self.sqrt_alphas_bar[t].view(-1, 1, 1, 1).to(device)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1, 1).to(device)
       
        print("sqrt_alpha_bar_t",sqrt_alpha_bar_t.shape)
        print("sqrt_one_minus_alpha_bar_t",sqrt_one_minus_alpha_bar_t.shape)
        # print("x_zero",x_zero.is_cuda)

        return sqrt_alpha_bar_t * x_zero + sqrt_one_minus_alpha_bar_t * noise


    def p_losses(self, denoise_model, x_zero, t, noise=None, loss_type="l1"):
        # TODO (2.2): compute the input to the network using the forward diffusion process and predict the noise using the model; if noise is None, you will need to create a new noise vector, otherwise use the provided one.
        if noise is None:
            noise = torch.randn_like(x_zero)

        x_t = self.q_sample(x_zero, t, noise)

        # Pass condition y to the model
        predicted_noise = denoise_model(x_t, t)

        if loss_type == 'l1':
            loss = F.l1_loss(predicted_noise, noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(predicted_noise, noise)
        else:
            raise NotImplementedError()

        return loss