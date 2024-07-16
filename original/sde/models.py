import jax
import jax.numpy as jnp
from markov_approximation import *
import jax.scipy.special as sp
import flax.linen as nn
import distrax
from taesd import TAESD
import pickle

TAE = False

class DiagonalNormal:
    def __init__(self, mu=jnp.zeros(1), sigma=jnp.ones(1), learnable=False):
        assert mu.shape == sigma.shape
        self._mu = mu
        self._sigma = sigma
        self.shape = mu.shape
        self.learnable = learnable

    def init(self, key):
        if self.learnable:
            return {
                'mu': self._mu,
                'logvar': 2 * jnp.log(self._sigma),
            }
        else:
            return {}

    def mu(self, params):
        if self.learnable:
            return params['mu']
        else:
            return self._mu

    def sigma(self, params):
        if self.learnable:
            return jnp.exp(.5 * params['logvar'])
        else:
            return self._sigma

    def logvar(self, params):
        if self.learnable:
            return params['logvar']
        else:
            return 2 * jnp.log(self._sigma)

    def sample(self, params, key):
        return jax.random.normal(key, self.shape) * self.sigma(params) + self.mu(params)

    def kl_divergence(self, params, other, other_params=None):
        mu_0 = self.mu(params)
        logvar_0 = self.logvar(params)
        mu_1 = other.mu(other_params)
        logvar_1 = other.logvar(other_params)
        return .5 * (jnp.exp(logvar_0 - logvar_1) - 1 + (mu_1 - mu_0) ** 2 / jnp.exp(logvar_1) + logvar_1 - logvar_0).sum()


class Beta:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    @property
    def mu(self):
        return self.alpha / (self.alpha + self.beta)

    @property
    def nu(self):
        return self.alpha + self.beta

    def sample(self, key):
        return jax.random.beta(key, self.alpha, self.beta)

    def kl_divergence(self, other):
        return sp.betaln(other.alpha, other.beta) - sp.betaln(self.alpha, self.beta) + (self.alpha - other.alpha) * sp.digamma(self.alpha) + (self.beta - other.beta) * sp.digamma(self.beta) + (other.alpha - self.alpha + other.beta - self.beta) * sp.digamma(self.alpha + self.beta)


def up(x):
    shape = x.shape
    new_shape = [*shape[:-3], 2 * shape[-3], 2 * shape[-2], shape[-1]]
    return jax.image.resize(x, new_shape, 'nearest')


class DownBlock(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.out_channels, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.max_pool(x, (2, 2), (2, 2))
        x = nn.GroupNorm(8)(x)
        x = nn.silu(x)
        return x


class UpBlock(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.out_channels, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.GroupNorm(8)(x)
        x = up(x)
        x = nn.silu(x)
        return x

'''
class Encoder(nn.Module):
    image_size: int
    num_channels: int
    num_features: int
    num_outputs: int

    @nn.compact
    def __call__(self, x):
        assert self.image_size == 64
        x = DownBlock(self.num_features)(x)
        x = DownBlock(2 * self.num_features)(x)
        x = DownBlock(4 * self.num_features)(x)
        x = nn.Conv(4,kernel_size=(3,3),padding='SAME')(x)
        return x


class Decoder(nn.Module):
    image_size: int
    num_channels: int
    num_features: int
    num_latents: int

    @nn.compact
    def __call__(self, x):
        x = UpBlock(4 * self.num_features)(x)
        x = UpBlock(2 * self.num_features)(x)
        x = UpBlock(self.num_features)(x)
        x = nn.Conv(self.num_features, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.silu(x)
        x = nn.Conv(self.num_channels, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.sigmoid(x)
        return x
'''


class Encoder(nn.Module):  ### Original Encoder  
    image_size: int
    num_channels: int
    num_features: int
    num_outputs: int

    @nn.compact
    def __call__(self, x):
        #assert self.image_size == 64
        x = DownBlock(self.num_features)(x)
        x = DownBlock(2 * self.num_features)(x)
        x = DownBlock(4 * self.num_features)(x)
        x = DownBlock(4 * self.num_features)(x)
        x_flat = x.reshape(x.shape[:-3] + (-1,))
        x = nn.Dense(self.num_outputs)(x_flat)
        return x


class Decoder(nn.Module):  ### Original Decoder
    image_size: int
    num_channels: int
    num_features: int
    num_latents: int
    num_contents: int

    @nn.compact
    def __call__(self, x):
        print(x.shape)
        x = nn.Dense(int(self.image_size/16 * self.image_size/16 * 4 * self.num_contents))(x)
        x = x.reshape(x.shape[:-1] + (int(self.image_size/16), int(self.image_size/16), 4*self.num_contents))
        x = UpBlock(4 * self.num_features)(x)
        x = UpBlock(2 * self.num_features)(x)
        x = UpBlock(self.num_features)(x)
        x = UpBlock(self.num_features)(x)
        x = nn.Conv(self.num_features, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.silu(x)
        x = nn.Conv(self.num_channels, kernel_size=(3, 3), padding='SAME')(x)
        print(x.shape)
        return x


class Content(nn.Module):       ### Original Content
    num_features: int
    num_contents: int
    num_content_frames: int

    @nn.compact
    def __call__(self, h):
        w = jnp.median(h[:self.num_content_frames], axis=-2)
        w = nn.Dense(self.num_contents)(w)
        w = nn.silu(w)
        w = nn.Dense(self.num_contents)(w)
        return w
    
class Infer(nn.Module):         ### Original Infer
    num_features: int
    num_latents: int

    @nn.compact
    def __call__(self, x):
        #num_frames, num_features = x.shape
        ##print(x.shape)
        h = nn.Conv(self.num_features, kernel_size=(3,), padding='SAME')(x)
        h = nn.silu(h)
        h = nn.Conv(self.num_features, kernel_size=(3,), padding='SAME')(h)

        g = nn.Dense(self.num_features)(jnp.concatenate([h[0], x[0], x[1], x[2]], axis=-1))
        g = nn.silu(g)
        g = nn.Dense(self.num_features)(g)
        g = nn.silu(g)
        g = nn.Dense(2 * self.num_latents)(g)

        x0_mean = g[:self.num_latents]
        x0_logvar = g[self.num_latents:]
        x0_posterior = distrax.MultivariateNormalDiag(x0_mean, jnp.exp(.5 * x0_logvar))
        return x0_posterior, h


# Spatial content block. (20, 8, 8, 4) -> (8, 8, 1)
class Content_spatial(nn.Module):
    num_features: int
    num_contents: int
    num_content_frames: int

    @nn.compact
    def __call__(self, h):
        w = jnp.median(h[:self.num_content_frames], axis=0)
        w = nn.Conv(4, kernel_size=(3, 3), padding='SAME')(w)
        w = nn.silu(w)
        w = nn.Conv(4, kernel_size=(3, 3), padding='SAME')(w)
        w = nn.silu(w)
        w = nn.Conv(4, kernel_size=(3, 3), padding='SAME')(w)
        #print("\nContent W shape:", w.shape,"\n")
        return w 

'''
class Content(nn.Module):
    num_features: int
    num_contents: int
    num_content_frames: int

    @nn.compact
    def __call__(self, h):
        w = h[:10]
        w = nn.Conv(16, kernel_size=(3, 3), padding='SAME')(w)
        w = nn.relu(w)
        w = nn.Conv(32, kernel_size=(3, 3), padding='SAME')(w)
        w = nn.relu(w)
        w = jnp.mean(w, axis=0)
        w = nn.Conv(4, kernel_size=(3, 3), padding='SAME')(w)
        w = nn.relu(w)
        w = nn.Conv(1, kernel_size=(3, 3), padding='SAME')(w)
        return w

'''

class Infer_spatial(nn.Module):
    num_features: int
    num_latents: int

    @nn.compact
    def __call__(self, x):
        #num_frames, num_features,_,_ = x.shape
        h = nn.Conv(4, kernel_size=(3,3), padding='SAME')(x)
        h = nn.silu(h)
        h = nn.Conv(4, kernel_size=(3,3), padding='SAME')(h)

        g = nn.Conv(8, kernel_size=(3,3), padding='SAME')(jnp.concatenate([h[0], x[0], x[1], x[2]], axis=-1))
        g = nn.silu(g)
        g = nn.Conv(1, kernel_size=(3,3), padding='SAME')(g)
        g = nn.silu(g)
        g = g.reshape(g.shape[:-3] + (-1,))
        print("G SHAPE: ", g.shape,"\n\n")
        g = nn.Dense(64)(g)
        g = nn.silu(g)
        g = nn.Dense(2 * self.num_latents)(g)

        x0_mean = g[:self.num_latents]
        x0_logvar = g[self.num_latents:]
        x0_posterior = distrax.MultivariateNormalDiag(x0_mean, jnp.exp(.5 * x0_logvar))
        return x0_posterior, h




class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(1000)(x)
        x = nn.tanh(x)
        x = nn.Dense(1000)(x)
        x = nn.tanh(x)
        x = nn.Dense(1)(x)
        x = nn.tanh(x)
        return x


class Function:
    def init(self, key):
        return {}

    def __call__(self, params, *args):
        raise NotImplementedError


class StaticFunction(Function):
    def __init__(self, function):
        self.function = function

    def init(self, key):
        return {}

    def __call__(self, params, *args):
        return self.function(*args)


class FractionalSDE:
    """
    Neural Stochastic Differential Equations driven by fractional Brownian Motion.
    Args:
        b (Function): The drift function of the SDE.
        u (Function): The control function of the SDE.
        s (Function): The diffusion function of the SDE.
        gamma (jnp.ndarray): The gamma values of the Ornstein-Uhlenbeck processes used to approximate fractional Brownian Motion.
        hurst (float or None): The Hurst coefficient of the fractional Brownian Motion. If None, then the Hurst coefficient is learnable. If -1, the model falls back to standard Brownian Motion (gamma = [0], omega = [1]).
        type [1, 2] (int): The type of the fractional Brownian Motion. 1 for type I, 2 for type II.
        time_horizon (float): The time horizon of the model, used to calculate omega.
        num_latents (int): The number of latent dimensions.
    """
    def __init__(
            self,
            b: Function,
            u: Function,
            s: Function,
            gamma: jnp.ndarray,
            hurst: float or None,
            type: int = 1,
            time_horizon: float = 1.,
            num_latents: int = 1,
        ):
        self.gamma = gamma
        self.type = type
        self.num_latents = num_latents
        self._b = b
        self._u = u
        self._s = s

        if type == 1:
            self.omega_fn = jax.jit(lambda hurst: omega_optimized_1(self.gamma, hurst, time_horizon))
        elif type == 2:
            self.omega_fn = jax.jit(lambda hurst: omega_optimized_2(self.gamma, hurst, time_horizon))
        else:
            raise ValueError('type must be either 1 or 2')

        if hurst is None:
            self._hurst = None
        elif hurst < 0:
            print('Falling back to standard Brownian Motion (gamma = [0], omega = [1]). Args gamma and type are ignored.')
            self._hurst = .5
            self.type = 2   # prevent problems with gamma = 0
            self.gamma = jnp.array([0.])
            self._omega = jnp.array([1.])
        else:
            self._hurst = hurst
            self._omega = self.omega_fn(hurst)

    @property
    def num_k(self):
        return len(self.gamma)

    def check_dt(self, dt):
        assert self.gamma.max() * dt < .5, 'dt too large for stable integration, please reduce dt or decrease largest gamma'

    def init(self, key):
        keys = jax.random.split(key, 3)
        params = {}

        if self._hurst is None:
            params['hurst_raw'] = 0.    # sigmoid(0.) = .5

        params['b'] = self._b.init(keys[0])
        params['u'] = self._u.init(keys[1])
        params['s'] = self._s.init(keys[2])
        return params

    def hurst(self, params):
        if self._hurst is None:
            return jax.nn.sigmoid(params['hurst_raw'])
        else:
            return self._hurst

    def omega(self, hurst):
        if self._hurst is None:
            return self.omega_fn(hurst)
        else:
            return self._omega

    def b(self, params, t, x, args):      # Prior drift.
        return self._b(params['b'], t, x, args)

    def u(self, params, t, x, y, args, pred=False):   # Approximate posterior control.
        return self._u(params['u'], t, x, y, args, pred)
    

    def s(self, params, t, x, args):      # Shared diffusion.
        return self._s(params['s'], t, x, args)

    def __call__(self, params, key, x0, ts, dt, solver='euler', pred = False, args=None):
        keys = jax.random.split(key, 4)

        hurst = self.hurst(params)
        omega = self.omega(hurst)

        if self.type == 1:
            cov = 1 / (self.gamma[None, :] + self.gamma[:, None])
            y0 = jax.random.multivariate_normal(keys[2], jnp.zeros((self.num_latents, self.num_k)), cov)
        elif self.type == 2:
            y0 = jnp.zeros((self.num_latents, self.num_k))

        if solver == 'euler':
            num_steps = int(jnp.ceil((ts[-1] - ts[0]) / dt))

            ts_, xs_, log_path = solve_vector(params, self, omega, x0, y0, ts[0], num_steps, dt, keys[3], args, pred)

            # interpolate for requested timesteps
            xs = jax.vmap(jnp.interp, in_axes=(None, None, 1), out_axes=1)(ts, ts_, xs_)
        else:

            xs, log_path = solve_diffrax(params, self, omega, x0, y0, ts, dt, keys[3], solver, pred, args)
        return xs, log_path


class Combine(nn.Module):
    content_dim: int
    latent_dim: int
    num_features: int

    @nn.compact
    def __call__(self, content, time):
       #print("\ncontent shape: ",content.shape,"\n")
        #print("\ntime shape: ",time.shape,"\n")
        content_transform = nn.Dense(self.num_features)(content)
        content_transform = nn.silu(content_transform)
        time_transform = nn.Dense(self.num_features)(time)
        time_transform = nn.silu(time_transform)
        combined = content_transform + time_transform
        combined = nn.Dense(self.num_features*4)(combined)
        combined = nn.silu(combined)
        #print("\ncombined1 shape: ",combined.shape,"\n")
        combined = jnp.reshape(combined, combined.shape[:-1] + (16, 16,1))
        #print("\ncombined2 shape: ",combined.shape,"\n")
        combined = nn.Conv(4, kernel_size=(3, 3), padding='SAME')(combined)
        combined = nn.silu(combined)
        combined = nn.Conv(4, kernel_size=(3, 3), padding='SAME')(combined)
        return combined


class Combine_sc(nn.Module):
    content_dim: int
    latent_dim: int
    num_features: int
    latent_size: int
    seq_len: int
    num_contents: int

    @nn.compact
    def __call__(self, content, time):
        #print("\ncontent shape: ",content.shape,"\n")
        #print("\ntime shape: ",time.shape,"\n")
        content = jnp.reshape(content, (1, self.latent_size, self.latent_size, 4))
        content = jnp.repeat(content,repeats=self.seq_len,axis=0)
        
        time_transform = nn.Dense(self.num_contents)(time)
        time_transform = nn.silu(time_transform)
        time_transform = jnp.reshape(time_transform, time_transform.shape[:-1] + (self.latent_size, self.latent_size, 1))
        combined = jnp.concatenate([content, time_transform], axis=-1)
        combined = nn.Dense(self.num_features)(combined)
        combined = nn.silu(combined)
        combined = nn.Conv(4, kernel_size=(3, 3), padding='SAME')(combined)
        return combined

class Linear1(nn.Module):
    num_contents: int
    @nn.compact
    def __call__(self, x):
        x = jnp.mean(x, axis=-1)
        x = jnp.reshape(x, (x.shape[0], self.num_contents))
        x = nn.Dense(self.num_contents)(x)
        return x
'''
class Linear2(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        return x

class Discriminator(nn.Module):
    num_features: int
    num_latents: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.num_features, kernel_size=(3,), padding='SAME')(x)
        x = nn.silu(x)
        x = nn.Conv(self.num_features, kernel_size=(3,), padding='SAME')(x)
        x = nn.silu(x)
        x = nn.Conv(self.num_features, kernel_size=(3,), padding='SAME')(x)
        x = nn.silu(x)
        x = nn.Dense(self.num_latents)(x)
        x = nn.silu(x)
        x = nn.Dense(1)(x)
        x = nn.sigmoid(x)
        return x


class Discriminator(nn.Module):
    num_features: int
    num_latents: int

    @nn.compact
    def __call__(self, x):

        assert x.shape == (25, 64), f"Expected input shape (25, 64), but got {x.shape}"
        
        x = nn.Conv(self.num_features, kernel_size=(3,), padding='SAME')(x)
        x = nn.silu(x)
        x = nn.Conv(self.num_features, kernel_size=(3,), padding='SAME')(x)
        x = nn.silu(x)
        x = nn.Conv(self.num_features, kernel_size=(3,), padding='SAME')(x)
        x = nn.silu(x)
        

        x = x.reshape(-1)
        
        x = nn.Dense(self.num_latents)(x)
        x = nn.silu(x)
        x = nn.Dense(1)(x)
        x = nn.sigmoid(x)
        
        return x

'''

class VideoSDE:
    """
    Latent Video Model.
    Args:
        x0_prior (diffrax.distribution.Distribution): Prior for x0.
        x0_prior_learnable (bool): Whether the prior for x0 is learnable.

    """
    def __init__(
        self,
        image_size,
        num_channels,
        num_features,
        num_latents,
        num_contents,
        x0_prior: distrax.Distribution,
        x0_prior_learnable: bool,
        sde,
        model_type: str = 'original',
    ):
        self.model_type = model_type
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_features = num_features
        self.num_latents = num_latents
        self.num_contents = num_contents
        self._x0_prior = x0_prior
        self.x0_prior_learnable = x0_prior_learnable
        self.latent_size = int(image_size/8)
        self.sequence_length = 20

        self.tae = True
        if model_type == 'original':
            self.tae = False
            self._encoder = Encoder(image_size, num_channels, num_features, int(num_contents/2))
            self._decoder = Decoder(image_size, num_channels, num_features, num_latents, int(num_contents/2))
            self._content = Content(num_features, int(num_contents/2), int(num_contents/2))
            self._infer = Infer(int(num_contents/2), num_latents)
            self._sde = sde
        elif model_type == 'plain_taesd':
            self._taesd = TAESD()
            self._content = Content(num_features, num_contents, num_contents)
            self._infer = Infer(num_features, num_latents)
            self._combine = Combine(num_features, num_latents, num_features)
            self._sde = sde
        elif model_type == 'sc_taesd':
            self._taesd = TAESD()
            self._content = Content_spatial(num_features, num_contents, 20)
            self._infer = Infer(num_features, num_latents)
            self._combine = Combine_sc(num_features, num_latents, num_features, self.latent_size, self.sequence_length, self.num_contents)
            self._sde = sde
        elif model_type == 'scsi_taesd':
            self._taesd = TAESD()
            self._content = Content_spatial(num_features, num_contents, 20)
            self._infer = Infer_spatial(num_features, num_latents)
            self._combine = Combine_sc(num_features, num_latents, num_features, self.latent_size, self.sequence_length, self.num_contents)
            self._linear1 = Linear1(num_contents)
            self._sde = sde


    def init(self, key):
        keys = jax.random.split(key, 5)
        params = {}

        if self.x0_prior_learnable:
            params['x0_prior'] = self._x0_prior

        dummy_num_timesteps = 5
        if self.model_type == 'original':
            params['encoder'] = self._encoder.init(keys[0], jnp.zeros((self.image_size, self.image_size, self.num_channels)))
            params['decoder'] = self._decoder.init(keys[1], jnp.zeros((dummy_num_timesteps, int(self.num_contents/2)+self.num_latents)))
            params['content'] = self._content.init(keys[2], jnp.zeros((dummy_num_timesteps, int(self.num_contents/2))))
            params['infer'] = self._infer.init(keys[3], jnp.zeros((dummy_num_timesteps, int(self.num_contents/2))))
            params['sde'] = self._sde.init(keys[4])
        elif self.model_type == 'plain_taesd':
            params['content'] = self._content.init(keys[0], jnp.zeros((dummy_num_timesteps, self.num_contents)))
            params['infer'] = self._infer.init(keys[1], jnp.zeros((dummy_num_timesteps, self.num_contents)))
            params['combine'] = self._combine.init(keys[2], jnp.zeros(self.num_features), jnp.zeros((30,self.num_latents)))
            with open('/kuacc/users/mkizil19/hpc_run/sdevideo/sdevideo/original/sde/taesd_flax_params.p', 'rb') as f:
                params_ = pickle.load(f)
            params['taesd'] = params_
            params['sde'] = self._sde.init(keys[3])
        elif self.model_type == 'sc_taesd':
            params['content'] = self._content.init(keys[0], jnp.zeros((dummy_num_timesteps, self.latent_size, self.latent_size, 4)))
            params['infer'] = self._infer.init(keys[1], jnp.zeros((dummy_num_timesteps, self.num_contents)))
            params['combine'] = self._combine.init(keys[2], jnp.zeros((self.latent_size,self.latent_size,4)), jnp.zeros((self.sequence_length,self.num_latents)))
            with open('/kuacc/users/mkizil19/hpc_run/sdevideo/sdevideo/original/sde/taesd_flax_params.p', 'rb') as f:
                params_ = pickle.load(f)
            params['taesd'] = params_
            params['sde'] = self._sde.init(keys[3])
        elif self.model_type == 'scsi_taesd':
            params['content'] = self._content.init(keys[0], jnp.zeros((dummy_num_timesteps, self.latent_size, self.latent_size, 4)))
            params['infer'] = self._infer.init(keys[1], jnp.zeros((dummy_num_timesteps, self.latent_size, self.latent_size, 4)))
            params['combine'] = self._combine.init(keys[2], jnp.zeros((self.latent_size, self.latent_size, 4)), jnp.zeros((self.sequence_length, self.num_latents)))
            with open('/kuacc/users/mkizil19/hpc_run/sdevideo/sdevideo/original/sde/taesd_flax_params.p', 'rb') as f:
                params_ = pickle.load(f)
            params['taesd'] = params_
            params['sde'] = self._sde.init(keys[3])
            params['linear1'] = self._linear1.init(keys[4], jnp.zeros((dummy_num_timesteps,self.latent_size ,self.latent_size,4)))
        


        return params
    
    #def linear2(self, params, *args):
     #   return self._linear2.apply(params['linear2'], *args)
    
    def linear1(self, params, *args):
        return self._linear1.apply(params['linear1'], *args)

    def x0_prior(self, params):
        if self.x0_prior_learnable:
            return params['x0_prior']
        else:
            return self._x0_prior

    def encoder(self, params, *args):
        if self.tae:
            return self._taesd.apply_encoder(params['taesd'], *args)
        else:
            return self._encoder.apply(params['encoder'], *args)

    def decoder(self, params, *args):
        if self.tae:    #### MODIFY ACCORINDG TO DATASETS CHANNEL
            if self.num_channels==1:
                return jnp.reshape(jnp.mean(self._taesd.apply_decoder(params['taesd'], *args),axis=-1),(self.sequence_length,self.image_size,self.image_size,1))
            else:
                return jnp.reshape(self._taesd.apply_decoder(params['taesd'], *args),(self.sequence_length,self.image_size,self.image_size,self.num_channels))
        else:
            return self._decoder.apply(params['decoder'], *args)
    
    def combine(self, params, content, time):
        return self._combine.apply(params['combine'], content, time)

    def content(self, params ,*args):
        return self._content.apply(params['content'], *args)

    def infer(self, params, *args):
        return self._infer.apply(params['infer'], *args)

    def sde(self, params, *args):
        return self._sde(params['sde'], *args)

    def __call__(self, params, key, ts, frames, dt, solver, pred=False):

        if self.model_type=='original':
            keys = jax.random.split(key, 2)
            x0_prior = self.x0_prior(params)
            print("Frames shape: ", frames.shape)
            h = self.encoder(params, frames)
            print("H shape: " , h.shape)
            w = self.content(params, h)
            print("W shape: ", w.shape)
            x0_posterior, h = self.infer(params, h)
            print("H shape: ", h.shape)
            x0 = x0_posterior.sample(seed=keys[0])
            print("X0 shape: ", x0.shape)
            kl_x0 = x0_posterior.kl_divergence(x0_prior)
            if pred:
                context = {'ts': ts[:10], 'hs': h}
            else:
                context = {'ts': ts, 'hs': h}
            xs, logpath = self.sde(params, keys[1], x0, ts, dt, solver, pred, {'context': context})
            frames_ = self.decoder(params,jnp.concatenate([w[None, :].repeat(len(xs), axis=0), xs], axis=-1))
            return frames_, (kl_x0, logpath)
        elif self.model_type=='plain_taesd':
            keys = jax.random.split(key, 2)
            if frames.shape[-1] == 1:
                frames = jnp.repeat(frames,repeats=3,axis=-1)
            print(frames.shape)
            num_frames, height, width, num_channels = frames.shape
            x0_prior = self.x0_prior(params)
            h = self.encoder(params, frames)
            print("H shape: ", h.shape)
            h = jnp.mean(h, axis=-1)
            h = h.reshape(num_frames, -1)
            w = self.content(params, h)
            print("W shape: ", w.shape)
            x0_posterior, h = self.infer(params, h)

            if pred:
                context = {'ts': ts[:10], 'hs': h}
            else:
                context = {'ts': ts, 'hs': h}
            x0 = x0_posterior.sample(seed=keys[0])
            kl_x0 = x0_posterior.kl_divergence(x0_prior)
            xs, logpath = self.sde(params, keys[1], x0, ts, dt, solver, pred,{'context': context})
            combined = self.combine(params, w, xs)
            frames_ = self.decoder(params, combined)
            return frames_, (kl_x0, logpath)
        elif self.model_type=='sc_taesd':
            keys = jax.random.split(key, 2)
            if frames.shape[-1] == 1:
                frames = jnp.repeat(frames,repeats=3,axis=-1)
            print("Frames shape: ", frames.shape)
            num_frames, height, width, num_channels = frames.shape
            x0_prior = self.x0_prior(params)
            h = self.encoder(params, frames)
            print("H shape: ", h.shape)
            w = self.content(params, h)
            print("W shape: ", w.shape)
            h = jnp.mean(h, axis=-1)
            h = h.reshape(num_frames, -1)
            x0_posterior, h = self.infer(params, h)
            if pred:
                context = {'ts': ts[:10], 'hs': h}
            else:
                context = {'ts': ts, 'hs': h}
            x0 = x0_posterior.sample(seed=keys[0])
            kl_x0 = x0_posterior.kl_divergence(x0_prior)
            xs, logpath = self.sde(params, keys[1], x0, ts, dt, solver, pred,{'context': context})
            combined = self.combine(params, w, xs)
            frames_ = self.decoder(params, combined)
            return frames_, (kl_x0, logpath)
        elif self.model_type=='scsi_taesd':
            keys = jax.random.split(key, 2)
            if frames.shape[-1] == 1:
                frames = jnp.repeat(frames,repeats=3,axis=-1)
            num_frames, height, width, num_channels = frames.shape
            x0_prior = self.x0_prior(params)
            print("Frames shape: ", frames.shape)
            h = self.encoder(params, frames)
            print("H shape: ", h.shape)
            w = self.content(params, h)
            print("W shape: ", w.shape)
            x0_posterior, h = self.infer(params, h)
            print("H shape: ", h.shape)
            h= self.linear1(params, h)
            if pred:
                context = {'ts': ts[:10], 'hs': h}
                print("TS shape: ", ts.shape)
                print("HS shape: ", h.shape)
            else:
                context = {'ts': ts, 'hs': h}
            x0 = x0_posterior.sample(seed=keys[0])
            kl_x0 = x0_posterior.kl_divergence(x0_prior)
            xs, logpath = self.sde(params, keys[1], x0, ts, dt, solver, pred,{'context': context})
            combined = self.combine(params, w, xs)
            frames_ = self.decoder(params, combined)
            return frames_, (kl_x0, logpath)

