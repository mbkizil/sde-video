import jax
import jax.numpy as jnp
import numpy as onp
import flax.linen as nn
import optax
import diffrax
import distrax
import equinox as eqx
from sde.markov_approximation import gamma_by_gamma_max
from sde.models_spatial_sde_and_content import FractionalSDE, VideoSDE
# from models import FractionalSDE, VideoSDE
import data
from sde.util import NumpyLoader
from tqdm import tqdm
import pickle
import typing
from jsonargparse import ArgumentParser
import wandb
import uuid
import imageio
import os


# class MLP(nn.Module):
#     num_outputs: int
#     activation: typing.Callable = lambda x: x

#     @nn.compact
#     def __call__(self, x):
#         x = nn.Dense(200)(x)
#         x = nn.tanh(x)
#         x = nn.Dense(200)(x)
#         x = nn.tanh(x)
#         x = nn.Dense(self.num_outputs)(x)
#         x = self.activation(x)
#         return x

class CNN(nn.Module):
    num_input_channels: int
    num_output_channels: int
    activation: typing.Callable = lambda x: x

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(16, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(16, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(1, kernel_size=(3, 3), padding="SAME")(x)
        return x

# class ControlFunction(eqx.Module):
#     layers: list

#     def __init__(self, key):
#         keys = jax.random.split(key, 4)
#         self.layers = [
#             eqx.nn.Conv2d(1, 16, kernel_size=3, padding=1, key=keys[0]),
#             jax.nn.relu,
#             eqx.nn.Conv2d(16, 16, kernel_size=3, padding=1, key=keys[1]),
#             jax.nn.relu,
#             eqx.nn.Conv2d(16, 16, kernel_size=3, padding=1, key=keys[2]),
#             jax.nn.relu,
#             eqx.nn.Conv2d(16, 1, kernel_size=3, padding=1, key=keys[3]),
#         ]

#     def __call__(self, t, x, args):
#         x_coord = jnp.linspace(-1, 1, x.shape[1])
#         y_coord = jnp.linspace(-1, 1, x.shape[2])
#         x_mesh, y_mesh = jnp.meshgrid(x_coord, y_coord)
#         x = jax.lax.concatenate([x, x_mesh, y_mesh], dimension=0)
#         for layer in self.layers:
#             x = layer(x)
#         return x

class ControlFunction:
    def __init__(self, num_k, num_latents, num_features):
        self.num_k = num_k
        self.num_latents = num_latents
        self.num_features = num_features
        self.cnn = CNN(1, 1)

    def init(self, key):
        params = self.cnn.init(key, jnp.zeros((self.num_latents, self.num_latents, 3)))
        # # Initialization trick from Glow.
        # params['params']['Dense_2']['kernel'] *= 0
        return params

    def __call__(self, params, t, x, *args):
        x = jnp.permute_dims(x, (1, 2, 0))
        x_coord = jnp.linspace(-1, 1, x.shape[0])
        y_coord = jnp.linspace(-1, 1, x.shape[1])
        x_mesh, y_mesh = jnp.meshgrid(x_coord, y_coord)
        x_mesh = jnp.expand_dims(x_mesh, axis=2)
        y_mesh = jnp.expand_dims(y_mesh, axis=2)
        x = jax.lax.concatenate([x, x_mesh, y_mesh], dimension=2)
        new_x = self.cnn.apply(params, x)
        return jnp.permute_dims(new_x, (2, 0, 1))
        # context = args['context']
        # h = jax.vmap(jnp.interp, (None, None, 1))(t, context['ts'], context['hs'])
        # return self.mlp.apply(params, jnp.concatenate([x, y.flatten(), h], axis=-1))
        
    
class Drift:
    def __init__(self, num_latents):
        self.num_latents = num_latents
        self.cnn = CNN(1, 1)

    def init(self, key):
        params = self.cnn.init(key, jnp.zeros((self.num_latents, self.num_latents, 3)))
        return params

    def __call__(self, params, t, x, *args):
        x = jnp.permute_dims(x, (1, 2, 0))
        x_coord = jnp.linspace(-1, 1, x.shape[0])
        y_coord = jnp.linspace(-1, 1, x.shape[1])
        x_mesh, y_mesh = jnp.meshgrid(x_coord, y_coord)
        x_mesh = jnp.expand_dims(x_mesh, axis=2)
        y_mesh = jnp.expand_dims(y_mesh, axis=2)
        x = jax.lax.concatenate([x, x_mesh, y_mesh], dimension=2)
        new_x = self.cnn.apply(params, x)
        return jnp.permute_dims(new_x, (2, 0, 1))
    
class Diffusion:
    # commutative noise!
    def __init__(self, num_latents):
        self.num_latents = num_latents
        self.cnn = CNN(1, 1)

    def init(self, key):
        params = self.cnn.init(key, jnp.zeros((self.num_latents, self.num_latents, 3)))
        return params

    def __call__(self, params, t, x, *args):
        x = jnp.permute_dims(x, (1, 2, 0))
        x_coord = jnp.linspace(-1, 1, x.shape[0])
        y_coord = jnp.linspace(-1, 1, x.shape[1])
        x_mesh, y_mesh = jnp.meshgrid(x_coord, y_coord)
        x_mesh = jnp.expand_dims(x_mesh, axis=2)
        y_mesh = jnp.expand_dims(y_mesh, axis=2)
        x = jax.lax.concatenate([x, x_mesh, y_mesh], dimension=2)
        new_x = self.cnn.apply(params, x)
        return jnp.permute_dims(new_x, (2, 0, 1))

# class Drift(eqx.Module):
#     layers: list

#     def __init__(self, key):
#         keys = jax.random.split(key, 4)
#         self.layers = [
#             eqx.nn.Conv2d(1, 16, kernel_size=3, padding=1, key=keys[0]),
#             jax.nn.relu,
#             eqx.nn.Conv2d(16, 16, kernel_size=3, padding=1, key=keys[1]),
#             jax.nn.relu,
#             eqx.nn.Conv2d(16, 16, kernel_size=3, padding=1, key=keys[2]),
#             jax.nn.relu,
#             eqx.nn.Conv2d(16, 1, kernel_size=3, padding=1, key=keys[3]),
#         ]

#     def __call__(self, t, x, args):
#         x_coord = jnp.linspace(-1, 1, x.shape[1])
#         y_coord = jnp.linspace(-1, 1, x.shape[2])
#         x_mesh, y_mesh = jnp.meshgrid(x_coord, y_coord)
#         x = jax.lax.concatenate([x, x_mesh, y_mesh], dimension=0)
#         for layer in self.layers:
#             x = layer(x)
#         return x

# class Diffusion(eqx.Module):
#     layers: list

#     def __init__(self, key):
#         keys = jax.random.split(key, 4)
#         self.layers = [
#             eqx.nn.Conv2d(1, 16, kernel_size=3, padding=1, key=keys[0]),
#             jax.nn.relu,
#             eqx.nn.Conv2d(16, 16, kernel_size=3, padding=1, key=keys[1]),
#             jax.nn.relu,
#             eqx.nn.Conv2d(16, 16, kernel_size=3, padding=1, key=keys[2]),
#             jax.nn.relu,
#             eqx.nn.Conv2d(16, 1, kernel_size=3, padding=1, key=keys[3]),
#         ]

#     def __call__(self, t, x, args):
#         x_coord = jnp.linspace(-1, 1, x.shape[1])
#         y_coord = jnp.linspace(-1, 1, x.shape[2])
#         x_mesh, y_mesh = jnp.meshgrid(x_coord, y_coord)
#         x = jax.lax.concatenate([x, x_mesh, y_mesh], dimension=-1)
#         for layer in self.layers:
#             x = layer(x)
#         return x
    
class SDE(eqx.Module):
    latent_shape: tuple
    prior_drift: eqx.Module
    diffusion: eqx.Module
    control: eqx.Module

    def __init__(self, key):
        keys = jax.random.split(key, 3)
        self.prior_drift = Drift(keys[0])
        self.diffusion = Diffusion(keys[1])
        self.control = ControlFunction(keys[2])
        self.latent_shape = (1, 8, 8)
        # self.latent_shape = (6, 6)

    def _drift(self, t, state, args):
        x, _ = state
        u = self.control(t, x, args)
        dx = self.prior_drift(t, x, args) + u
        return dx, .5 * u ** 2

    def _diffusion(self, t, state, args):
        x, _ = state
        return self.diffusion(t, x, args), jnp.zeros(self.latent_shape)

    def __call__(self, key, x0, ts, dt):
        keys = jax.random.split(key, 2)
        state_init = (x0, jnp.zeros(self.latent_shape))
        
        brownian_motion = diffrax.VirtualBrownianTree(ts[0], ts[-1], dt, (jax.ShapeDtypeStruct(self.latent_shape, x0.dtype), jax.ShapeDtypeStruct(self.latent_shape, x0.dtype)), keys[0])
        terms = diffrax.MultiTerm(diffrax.ODETerm(self._drift), diffrax.WeaklyDiagonalControlTerm(self._diffusion, brownian_motion))
        solution = diffrax.diffeqsolve(
            terms,
            diffrax.EulerHeun(),
            ts[0],
            ts[-1],
            dt0=dt,
            y0=state_init,
            saveat=diffrax.SaveAt(ts=ts),
        )
        xs, kl_path_int = solution.ys
        kl = kl_path_int[-1]  # the kl intergral was computed along with the solve, so the final value is what we need, see output of _drift above
        return xs, kl


def build_data_and_model(
        dataset: str,
        white: bool,
        num_latents: int,
        num_contents: int,
        num_features: int,
        num_k: int,
        gamma_max: float,
        int_sub_steps: int,
    ):

    if white:
        num_k = 1
        gamma = None
        hurst = - 1
    else:
        gamma = gamma_by_gamma_max(num_k, gamma_max)
        hurst = None

    data_train, data_val, dataset_kwargs = data.get(dataset)
    ts = jnp.arange(len(data_train[0])) * dataset_kwargs['dt']
    dt = dataset_kwargs['dt'] / int_sub_steps

    key = jax.random.PRNGKey(0)
    b = Drift(num_latents)
    u = ControlFunction(num_k, num_latents, num_features)
    # u = ControlFunction(key)
    s = Diffusion(num_latents)
    sde = FractionalSDE(b, u, s, gamma, hurst=hurst, type=1, time_horizon=ts[-1], num_latents=num_latents)
    latent_shape = (8, 8)
    # sde = SDE(key)
    x0_prior = distrax.MultivariateNormalDiag(jnp.zeros(num_latents), jnp.ones(num_latents))
    model = VideoSDE(dataset_kwargs['image_size'], dataset_kwargs['num_channels'], num_features, num_latents, num_contents, x0_prior, True, sde)
    # model._sde.check_dt(dt)
    params = model.init(key)
    return ts, dt, data_train, data_val, model, params


def train(
        dataset: str,
        white: bool = False,    # fallback to standard sde
        batch_size: int = 32,
        num_epochs: int = 100,
        num_latents: int = 4,
        # num_contents: int = 6,
        # num_features: int = 8,
        num_contents: int = 64,
        num_features: int = 64,
        num_k: int = 5,
        gamma_max: float = 20.,
        int_sub_steps: int = 3,
        kl_weight: float = 1.,
        log_video_interval: int = 100,
        use_wandb: bool = True
    ):
    solver = diffrax.StratonovichMilstein()

    ts, dt, data_train, data_val, model, params= build_data_and_model(dataset, white, num_latents, num_contents, num_features, num_k, gamma_max, int_sub_steps)
    dataloader = NumpyLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

    def loss_fn(params, key, frames):
        frames_, (kl_x0, logpath) = model(params, key, ts, frames, dt, solver)
        nll = ((frames - frames_) ** 2).sum()
        loss = nll + kl_weight * (kl_x0 + logpath)
        return loss, (nll, kl_x0, logpath)

    def batched_loss_fn(params, key, frames, batch_size=batch_size):
        keys = jax.random.split(key, batch_size)
        loss, aux = jax.vmap(loss_fn, (None, 0, 0))(params, keys, frames)
        return loss.mean(), jax.tree_util.tree_map(jnp.mean, aux)

    loss_grad = jax.jit(jax.value_and_grad(batched_loss_fn, has_aux=True))
    
#     partition_optimizers = {'trainable': optax.adam(3e-4), 'frozen': optax.set_to_zero()}
#     param_partitions = traverse_util.path_aware_map(lambda path, v: 'frozen' if 'taesd' in path else 'trainable', params)
#     tx = optax.multi_transform(partition_optimizers, param_partitions)
    
#     print(param_partitions)
    # params = eqx.filter(model, eqx.is_array)
    optimizer = optax.adam(3e-4)
    print(params.keys())
    opt_state = optimizer.init(params)
    random_key = jax.random.PRNGKey(7)
    pri = False
    seco = False
    for epoch in range(num_epochs):
        pbar = tqdm(range(len(dataloader)))
        for step, frames in zip(pbar, dataloader):
            random_key, key = jax.random.split(random_key)
            prm = params['taesd']
            (loss, loss_aux), grads = loss_grad(params, key, frames)
            nll, kl_x0, logpath = loss_aux
            updates, opt_state = optimizer.update(grads, opt_state)
            if pri:
                print("\n GRADS \n")
                print(grads)
                print("\n UPDATES \n")
                print(updates)
                if seco:
                    pri = False
                seco = True
            params = optax.apply_updates(params, updates)
            params['taesd'] = prm
            pbar.set_description(f'[Epoch {epoch+1}/{num_epochs}] Loss: {float(loss):.2f}, Hurst: {model._sde.hurst(params["sde"]):.2f}, NLL: {nll:.2f}, KL_x0: {kl_x0:.2f}, KL_path: {logpath:.2f}')

            if onp.isnan(float(loss)):
                return

            if use_wandb:
                wandb.log({
                    'loss': float(loss),
                    'nll': float(nll),
                    'kl_x0': float(kl_x0),
                    'kl_path': float(logpath),
                    'hurst': float(model._sde.hurst(params["sde"])),
                })
            
                if step % log_video_interval == 0 and use_wandb:
                    log_gif(model, frames, params, key, ts, dt, solver, (0, 1), "video_gif")

        with open('params.p', 'wb') as f:
            pickle.dump(params, f)

        if use_wandb:
            wandb.save('params.p')

    if use_wandb:
        wandb.join(quiet=True)
    
def log_gif(model, frames, params, key, ts, dt, solver, range, name):
    frames = jax.lax.stop_gradient(frames)
    # print(f"Frames shape: {frames.shape}")
    video, _ = model(params, key, ts, frames[0], dt, solver)
    # print(f"Generated video shape: {video.shape}")
    filename = f"tmp/{str(uuid.uuid4())}.gif"
    filename_recon = f"tmp/{str(uuid.uuid4())}_recon.gif"
    with imageio.get_writer(filename, mode="I") as writer:
        original_video = frames[0]
        original_video = jnp.tile(original_video, (1, 1, 1, 3))
        for b_frame in original_video:
            frame = (b_frame * 255).astype(jnp.uint8)
            writer.append_data(frame)
    
    with imageio.get_writer(filename_recon, mode="I") as writer:
        video = jnp.tile(video, (1, 1, 1, 3))
        for b_frame in video:
            frame = (b_frame * 255).astype(jnp.uint8)
            writer.append_data(frame)
    
    wandb.log({
        name: wandb.Video(filename, fps=2, format="gif"),
        name+"_recon": wandb.Video(filename_recon, fps=2, format="gif")
    })
    os.remove(filename)
    os.remove(filename_recon)


if __name__ == '__main__':
    print(jax.default_backend())
    parser = ArgumentParser()
    parser.add_function_arguments(train, as_positional=False)

    cfg = parser.parse_args()
    if cfg.use_wandb:
        wandb.init(project=f'jax-new-{cfg.dataset}', config=cfg)
    train(**cfg)
