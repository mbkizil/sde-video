import jax
import jax.numpy as jnp
import numpy as onp
import flax.linen as nn
import optax
import diffrax
import distrax
from markov_approximation import gamma_by_gamma_max
from models import FractionalSDE, VideoSDE
import data
from util import NumpyLoader
from tqdm import tqdm
import pickle
import typing
from jsonargparse import ArgumentParser
import wandb
import uuid
import imageio
import os


class MLP(nn.Module):
    num_outputs: int
    activation: typing.Callable = lambda x: x

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(200)(x)
        x = nn.tanh(x)
        x = nn.Dense(200)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.num_outputs)(x)
        x = self.activation(x)
        return x


class ControlFunction:
    def __init__(self, num_k, num_latents, num_features):
        self.num_k = num_k
        self.num_latents = num_latents
        self.num_features = num_features
        self.mlp = MLP(num_latents)

    def init(self, key):
        params = self.mlp.init(key, jnp.zeros(self.num_latents * (self.num_k + 1) + self.num_features))
        # Initialization trick from Glow.
        params['params']['Dense_2']['kernel'] *= 0
        return params

    def __call__(self, params, t, x, y, args):
        context = args['context']
        h = jax.vmap(jnp.interp, (None, None, 1))(t, context['ts'], context['hs'])
        return self.mlp.apply(params, jnp.concatenate([x, y.flatten(), h], axis=-1))


class Drift:
    def __init__(self, num_latents):
        self.num_latents = num_latents
        self.mlp = MLP(num_latents)

    def init(self, key):
        params = self.mlp.init(key, jnp.zeros(self.num_latents))
        return params

    def __call__(self, params, t, x, args):
        return self.mlp.apply(params, x)


class Diffusion:
    # commutative noise!
    def __init__(self, num_latents):
        self.num_latents = num_latents
        self.mlp = MLP(1, nn.softplus)

    def init(self, key):
        keys = jax.random.split(key, self.num_latents)
        params = jax.vmap(self.mlp.init)(keys, jnp.zeros((self.num_latents, 1)))
        return params

    def __call__(self, params, t, x, args):
        return jax.vmap(self.mlp.apply)(params, x[:, None])[:, 0]


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
    s = Diffusion(num_latents)
    sde = FractionalSDE(b, u, s, gamma, hurst=hurst, type=1, time_horizon=ts[-1], num_latents=num_latents)
    x0_prior = distrax.MultivariateNormalDiag(jnp.zeros(num_latents), jnp.ones(num_latents))
    model = VideoSDE(dataset_kwargs['image_size'], dataset_kwargs['num_channels'], num_features, num_latents, num_contents, x0_prior, True, sde)
    model._sde.check_dt(dt)
    params = model.init(key)
    return ts, dt, data_train, data_val, model, params


def train(
        dataset: str,
        white: bool = False,    # fallback to standard sde
        batch_size: int = 32,
        num_epochs: int = 100,
        num_latents: int = 4,
        num_contents: int = 64,
        num_features: int = 64,
        num_k: int = 5,
        gamma_max: float = 20.,
        int_sub_steps: int = 3,
        kl_weight: float = 1.,
        log_video_interval: int = 1000
    ):
    solver = diffrax.StratonovichMilstein()

    ts, dt, data_train, data_val, model, params = build_data_and_model(dataset, white, num_latents, num_contents, num_features, num_k, gamma_max, int_sub_steps)
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

    optimizer = optax.adam(3e-4)
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

            wandb.log({
                'loss': float(loss),
                'nll': float(nll),
                'kl_x0': float(kl_x0),
                'kl_path': float(logpath),
                'hurst': float(model._sde.hurst(params["sde"])),
            })
            
            if step % log_video_interval == 0:
                log_gif(model, frames, params, key, ts, dt, solver, (0, 1), "video_gif")

        with open('params.p', 'wb') as f:
            pickle.dump(params, f)
        wandb.save('params.p')

    wandb.join(quiet=True)
    
def log_gif(model, frames, params, key, ts, dt, solver, range, name):
    video, _ = model(params, key, ts, frames[0], dt, solver)
    video = jnp.repeat(jnp.array(video), 3, axis=-1)
    filename = f"tmp/{str(uuid.uuid4())}.gif"
    with imageio.get_writer(filename, mode="I") as writer:
        for b_frame in video:
            frame = (b_frame * 255).astype(jnp.uint8)
            writer.append_data(frame)

    wandb.log({
        name: wandb.Video(filename, fps=2, format="gif")
    })
    os.remove(filename)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_function_arguments(train, as_positional=False)

    cfg = parser.parse_args()
    wandb.init(project=f'jax-new-{cfg.dataset}', config=cfg)
    train(**cfg)
