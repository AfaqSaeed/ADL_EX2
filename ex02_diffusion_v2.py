import torch
import torch.nn.functional as F
from ex02_helpers import extract
from tqdm import tqdm
import matplotlib.pyplot as plt

def plot_beta_schedulers(timesteps):
    linear_betas = linear_beta_schedule(0.0001, 0.02,timesteps)
    cosine_betas = cosine_beta_schedule(timesteps)
    sigmoid_betas = sigmoid_beta_schedule(0.0001, 0.2,timesteps)
    
    plt.figure(figsize=(10, 6))
    plt.plot(linear_betas, label='Linear Schedule')
    plt.plot(sigmoid_betas, label='Sigmoid Schedule')
    plt.plot(cosine_betas[:-1], label='Cosine Schedule')  # Ensure same length for cosine

    plt.xlabel("Timesteps")
    plt.ylabel("Beta (Noise Variance)")
    plt.title("Comparison of Beta Schedules")
    plt.legend()
    plt.grid()
    plt.show()

def linear_beta_schedule(beta_start, beta_end, timesteps):
    """
    standard linear beta/variance schedule as proposed in the original paper
    """
    return torch.linspace(beta_start, beta_end, timesteps)


# # TODO: Transform into task for students
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    also read:
    https://www.analyticsvidhya.com/blog/2024/07/noise-schedules-in-stable-diffusion/
    """
    # TODO (2.3): Implement cosine beta/variance schedule as discussed in the paper mentioned above
    steps = torch.linspace(0, timesteps, timesteps+1 , dtype=torch.float32)  # Exclude endpoint
    steps = steps/timesteps
    alphas = torch.cos(((steps  + s) / (1 + s)) * torch.pi*0.5) ** 2    
    alphas = alphas / alphas[0]
    betas = 1 - (alphas[1:] / alphas[:-1])
    betas = torch.clamp(betas, max=0.02)
    return betas

def sigmoid_beta_schedule(beta_start, beta_end, timesteps):
    """
    sigmoidal beta schedule - following a sigmoid function
    """
    # TODO (2.3): Implement a sigmoidal beta schedule. Note: identify suitable limits of where you want to sample the sigmoid function.
    # Note that it saturates fairly fast for values -x << 0 << +x
    clip_min=1e-9
    steps = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float32)
    steps = 2*steps/timesteps
    start = torch.tensor(beta_start).sigmoid()
    end = torch.tensor(beta_end).sigmoid()
    eq = ((steps * (beta_end - beta_start) + (beta_start-beta_end))).sigmoid()
    eq =  (end - eq) / (end - start)
    alphas = eq/eq[0]
    betas = 1 - (alphas[1:]/alphas[:-1])
    betas = torch.clip(betas, clip_min, 0.999)
    return betas



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

        # TODO (2.2): Compute the central values for the equation in the forward pass already here so you can quickly
        #  use them in the forward pass.
        # Note that the function torch.cumprod may be of help
        # define alphas
        alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)


    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        # TODO (2.2): implement the reverse diffusion process of the model for (noisy) samples x and timesteps t.
        #  Note that x and t both have a batch dimension

        
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * model.predict(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        # TODO (2.2): The method should return the image at timestep t-1.
        if t_index == 0:
            return model_mean
        else:
            # posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            posterior_variance_t = betas_t
            noise = torch.randn_like(x)
            image_t_1 = model_mean + torch.sqrt(posterior_variance_t) * noise
            return image_t_1

    # Algorithm 2 (including returning all images)
    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=3, device="cuda"):
        # TODO (2.2): Implement the full reverse diffusion loop from random noise to an image, iteratively ''reducing''
        #  the noise in the generated image.
        # device = next(model.parameters()).device
        img = torch.randn((batch_size, channels, image_size, image_size), device=device)
        imgs = [img]

        for t in tqdm(reversed(range(0, self.timesteps)), total=self.timesteps):
            img = self.p_sample(model, img, torch.full((batch_size,), t, device=device, dtype=torch.long), t)
            imgs.append(img)

        return imgs

    def q_sample(self, x_zero, t, noise=None,device="cuda"):
        # TODO (2.2): Implement the forward diffusion process using the beta-schedule defined in the constructor;
        #  if noise is None, you will need to create a new noise vector, otherwise use the provided one.
        if noise is None:
            noise = torch.randn_like(x_zero)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_zero.shape).to(device)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_zero.shape
        ).to(device)
        noisy_image = sqrt_alphas_cumprod_t * x_zero + sqrt_one_minus_alphas_cumprod_t * noise

        return noisy_image

    def p_losses(self, denoise_model, x_zero, t, classes, noise=None, loss_type="l1"):
        # TODO (2.2): compute the input to the network using the forward diffusion process and predict the noise using
        #  the model; if noise is None, you will need to create a new noise vector, otherwise use the provided one.
        if noise is None:
            noise = torch.randn_like(x_zero,dtype=torch.float32)
        
        x_t = self.q_sample(x_zero, t, noise).to(torch.float32)

        predicted_noise = denoise_model(x_t, t,classes=classes)

        if loss_type == 'l1':
            # TODO (2.2): implement an L1 loss for this task
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            # TODO (2.2): implement an L2 loss for this task
            loss = F.mse_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()
        return loss

if __name__=="__main__":
    plot_beta_schedulers(100)   
    # Parameters
    