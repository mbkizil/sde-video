import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import optax
import diffrax
import distrax
from markov_approximation import gamma_by_gamma_max
from models import FractionalSDE, VideoSDE
import data
from tqdm import tqdm
import pickle
import typing
from jsonargparse import ArgumentParser
from taesd import TAESD

from PIL import Image
import os
from util import NumpyLoader



def up(x):
    shape = x.shape
    new_shape = [*shape[:-3], 2 * shape[-3], 2 * shape[-2], shape[-1]]
    return jax.image.resize(x, new_shape, 'nearest')


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
    print(dataset_kwargs)
    ts = jnp.arange(len(data_train[0])) * dataset_kwargs['dt']
    dt = dataset_kwargs['dt'] / int_sub_steps

    key = jax.random.PRNGKey(0)
    x0_prior = distrax.MultivariateNormalDiag(jnp.zeros(num_latents), jnp.ones(num_latents))
    with open('sde/taesd_flax_params.p', 'rb') as f:
        params_ = pickle.load(f)
    return ts, dt, data_train, data_val, params_


def train(
        dataset: str,
        white: bool = False,    # fallback to standard sde
        batch_size: int = 1,
        num_epochs: int = 100,
        num_latents: int = 4,
        num_contents: int = 64,
        num_features: int = 64,
        num_k: int = 5,
        gamma_max: float = 20.,
        int_sub_steps: int = 3,
        kl_weight: float = 1.,
    ):

    ts, dt, data_train, data_val,params_ = build_data_and_model(dataset, white, num_latents, num_contents, num_features, num_k, gamma_max, int_sub_steps)
    dataloader = NumpyLoader(data_train, batch_size=1, shuffle=True, num_workers=8, drop_last=True)
    #dataloader2 = NumpyLoader(data_val, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

    #os.makedirs("frames", exist_ok=True)
    #os.makedirs("latents", exist_ok=True)
    #os.makedirs("recons", exist_ok=True)


    random_key = jax.random.PRNGKey(7)
    latent_vectors = []
    pbar = tqdm(range(len(dataloader)))
    taesd = TAESD()
    i=0
    for step, frames in zip(pbar, dataloader):
        frames = frames[0]
        #print("\n\n\n")
        #print(frames[0].max())
        #print("\n\n\n")
        frames = jnp.repeat(frames,repeats=3,axis=-1)
        #img = Image.fromarray(np.array(255*frames[0],dtype=np.uint8))
        #img.save(f"frames/frame_{i}_{0}.png")

        random_key, key = jax.random.split(random_key)
        latents = taesd.apply_encoder(params_, frames)
        recons = taesd.apply_decoder(params_,latents)
        print(latents.shape)
        #print(latents.shape)
        #latent_imgo =latents[0]
        #latent_img = (latent_img - latent_img.min()) / (latent_img.max() - latent_img.min())
        #latent_imgo = (latent_imgo - latent_imgo.min()) / (latent_imgo.max() - latent_imgo.min())
        #latent_img = Image.fromarray(np.array(latent_imgo[:,:,0] * 255,dtype=np.uint8))
        #latent_img.save(f"latents/latent_{i}_{0}.png")
        #latent_img = Image.fromarray(np.array(latent_imgo[:,:,1] * 255,dtype=np.uint8))
        #latent_img.save(f"latents/latent_{i}_{1}.png")
        #latent_img = Image.fromarray(np.array(latent_imgo[:,:,2] * 255,dtype=np.uint8))
        #latent_img.save(f"latents/latent_{i}_{2}.png")
        #latent_img = Image.fromarray(np.array(latent_imgo[:,:,3] * 255,dtype=np.uint8))
        #latent_img.save(f"latents/latent_{i}_{3}.png")

        #latent_vectors.append(latents)
        print(recons[0].max())
        print(recons.shape)
        recons = recons.mean(axis=-1)
        recons = recons.clip(0,1)
        recon = Image.fromarray(np.array(recons[0] * 255,dtype=np.uint8))
        recon.save(f"recons/recon_{i}_{0}.png")
        i+=1
        if i>3:
            break
 

            
    ##latent_vectors = jnp.concatenate(latent_vectors, axis=0)
    ##jnp.save("latent_norm.npy",latent_vectors)






if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_function_arguments(train, as_positional=False)

    cfg = parser.parse_args()
    train(**cfg)
