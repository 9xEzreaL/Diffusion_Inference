import math
import torch
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from core.base_network import BaseNetwork


class Network(BaseNetwork):
    def __init__(self, unet, beta_schedule, module_name='sr3', new_sample_step=2000, eta=1.0 , **kwargs):
        super(Network, self).__init__(**kwargs)
        if module_name == 'sr3':
            from .sr3_modules.unet import UNet
        elif module_name == 'guided_diffusion':
            from .guided_diffusion_modules.unet import UNet

        self.denoise_fn = UNet(**unet)
        self.beta_schedule = beta_schedule
        self.new_sample_step=new_sample_step
        self.eta=eta

    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn

    def set_new_noise_schedule(self, device=torch.device('cuda'), phase='train'):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(**self.beta_schedule[phase])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        gammas = np.cumprod(alphas, axis=0)
        gammas_prev = np.append(1., gammas[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('gammas', to_torch(gammas))
        self.register_buffer('sqrt_recip_gammas', to_torch(np.sqrt(1. / gammas)))
        self.register_buffer('sqrt_recipm1_gammas', to_torch(np.sqrt(1. / gammas - 1)))

        self.register_buffer('gammas_prev', to_torch(gammas_prev))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - gammas_prev) / (1. - gammas)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(gammas_prev) / (1. - gammas)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - gammas_prev) * np.sqrt(alphas) / (1. - gammas)))


    def predict_start_from_noise(self, y_t, t, noise):
        return (
                extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t -
                extract(self.sqrt_recipm1_gammas, t, y_t.shape) * noise
        )

    def q_posterior(self, y_0_hat, y_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, y_t.shape) * y_0_hat +
                extract(self.posterior_mean_coef2, t, y_t.shape) * y_t
        )
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, y_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    def q_sample(self, y_0, sample_gammas, noise=None):
        noise = default(noise, lambda: torch.randn_like(y_0))
        return (
                sample_gammas.sqrt() * y_0 +
                (1 - sample_gammas).sqrt() * noise
        )

    @torch.no_grad()
    def resampling_timestep_and_parameters(self, device=torch.device('cuda'), phase='test', new_step=400):
        assert self.num_timesteps % new_step == 0, 'num_timesteps / new_timestep must ==0'
        self.sample_interval = int(self.num_timesteps / new_step)
        self.num_timesteps = new_step

        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        # Maybe this is right, but just by experiment result, don't know why...
        gammas = self.gammas.detach().cpu().numpy() if isinstance(
            self.gammas, torch.Tensor) else self.gammas
        last_gammas = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(gammas):
            if i % self.sample_interval == 0:
                new_betas.append(1 - alpha_cumprod / last_gammas)
                last_gammas = alpha_cumprod
        betas = np.array(new_betas)

        alphas = 1. - betas

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        gammas = np.cumprod(alphas, axis=0)
        gammas_prev = np.append(1., gammas[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('gammas', to_torch(gammas))
        self.register_buffer('sqrt_one_minus_gammas', to_torch(np.sqrt(1-gammas)))
        self.register_buffer('sqrt_recip_gammas', to_torch(np.sqrt(1. / gammas)))
        self.register_buffer('sqrt_recipm1_gammas', to_torch(np.sqrt(1. / gammas - 1)))

        self.register_buffer('gammas_prev', to_torch(gammas_prev))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - gammas_prev) / (1. - gammas)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(gammas_prev) / (1. - gammas)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - gammas_prev) * np.sqrt(alphas) / (1. - gammas)))


    @torch.no_grad()
    def make_ddim_sampling_parameters(self, y_t, t):
        # sigmas = eta * ((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
        # return sigmas, alphas, alphas_prev
        alphas = extract(self.gammas, t, y_t.shape)
        alphas_prev = extract(self.gammas_prev, t, y_t.shape)

        sigmas = self.eta * ((1 - alphas_prev) / (1 - alphas)).sqrt() * ((1 - alphas / alphas_prev)).sqrt()
        return sigmas, alphas, alphas_prev

    @torch.no_grad()
    def p_sample(self, y_t, t, t_next, clip_denoised=True, y_cond=None, guided_step=0, guided_mask=None, task='inpaint'):
        b, *_, device = *y_t.shape, y_t.device
        sigmas, alphas, alphas_prev = self.make_ddim_sampling_parameters(y_t, t)
        sqrt_one_minus_alphas = extract(self.sqrt_one_minus_gammas, t, y_t.shape)

        def get_model_output(y_cond, y_t, t):
            noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
            if task in ['cond', 'inpainting', 'inpaint']:
                pred_noise = self.denoise_fn(torch.cat([y_cond, y_t], dim=1), noise_level)
            else:
                pred_noise = self.denoise_fn(y_t, noise_level)
            return pred_noise

        def get_x_prev_and_pred_x0(e_t):
            a_t = alphas
            a_prev = alphas_prev
            sigma_t = sigmas
            sqrt_one_minus_at = sqrt_one_minus_alphas
            # current prediction for x_0
            pred_x0 = (y_t - sqrt_one_minus_at * e_t) / a_t.sqrt()
            if guided_mask is not None and t[0] > guided_step:
                pred_x0 = y_cond * (1 - guided_mask) + pred_x0 * guided_mask

            # direction pointing to x_t
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            # noise = sigma_t * noise_like(y_t.shape, device, repeat_noise) # * temperature
            noise = sigmas * torch.randn_like(y_t) if any(t > 0) else torch.zeros_like(y_t)

            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
            return x_prev, pred_x0

        e_t = get_model_output(y_cond, y_t, t)
        if len(self.old_eps) == 0:
            # Pseudo Improved Euler (2nd order)
            x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t)
            e_t_next = get_model_output(y_cond, x_prev, t_next)
            e_t_prime = (e_t + e_t_next) / 2
        elif len(self.old_eps) == 1:
            # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (3 * e_t - self.old_eps[-1]) / 2
        elif len(self.old_eps) == 2:
            # 3nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (23 * e_t - 16 * self.old_eps[-1] + 5 * self.old_eps[-2]) / 12
        elif len(self.old_eps) >= 3:
            # 4nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (55 * e_t - 59 * self.old_eps[-1] + 37 * self.old_eps[-2] - 9 * self.old_eps[-3]) / 24

        x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t_prime)

        return x_prev, pred_x0, e_t

    @torch.no_grad()
    def restoration(self, y_cond, y_t=None, y_0=None, mask=None, sample_num=8, guided_step=0, guided_mask=None,
                    task='inpaint'):
        b, *_ = y_cond.shape
        self.resampling_timestep_and_parameters(new_step=self.new_sample_step)
        self.old_eps = [] # represent old predicted noise

        assert self.num_timesteps > sample_num, 'num_timesteps must greater than sample_num'
        sample_inter = (self.num_timesteps // sample_num)

        y_t = default(y_t, lambda: torch.randn_like(y_cond))
        ret_arr = y_t
        time_range = list(reversed(range(0, self.num_timesteps)))
        for i in tqdm(time_range, desc='sampling loop time step', total=self.num_timesteps):
            t = torch.full((b,), i, device=y_cond.device, dtype=torch.long)
            t_next = torch.full((b,), time_range[min(i+1, len(time_range)-1)], device=y_cond.device, dtype=torch.long)
            y_t, pred_y_0, pred_noise = self.p_sample(y_t, t, t_next, y_cond=y_cond, guided_step=guided_step, guided_mask=guided_mask,
                                          task=task)
            self.old_eps.append(pred_noise)
            if mask is not None:
                y_t = y_0 * (1. - mask) + mask * y_t
                pred_y_0 = y_0 * (1. - mask) + mask * pred_y_0
            if i % sample_inter == 0:
                ret_arr = torch.cat([ret_arr, y_t], dim=0) # y_t
            if len(self.old_eps) >= 4:
                self.old_eps.pop(0)

        return y_t, ret_arr

    def forward(self, y_0, y_cond=None, mask=None, noise=None):
        # y0 : gt image
        # y cond : noise in hole image
        # mask : inside hole == 1
        # sampling from p(gammas)
        b, *_ = y_0.shape
        t = torch.randint(1, self.num_timesteps, (b,), device=y_0.device).long()
        gamma_t1 = extract(self.gammas, t - 1, x_shape=(1, 1))
        sqrt_gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))
        sample_gammas = (sqrt_gamma_t2 - gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1
        sample_gammas = sample_gammas.view(b, -1)

        noise = default(noise, lambda: torch.randn_like(y_0))
        y_noisy = self.q_sample(
            y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise)

        if mask is not None:
            noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy * mask + (1. - mask) * y_0], dim=1), sample_gammas)
            loss = self.loss_fn(mask * noise, mask * noise_hat)
        else:
            noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy], dim=1), sample_gammas)
            loss = self.loss_fn(noise, noise_hat)
        return loss


# gaussian diffusion trainer class
def exists(x):
    return x is not None


def default(val, d):
    if val is not None:
    # if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape=(1, 1, 1, 1)):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# beta_schedule function
def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-6, linear_end=1e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) /
                n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


