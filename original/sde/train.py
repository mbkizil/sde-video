from traceback import FrameSummary
import jax
import jax.numpy as jnp
import numpy as onp
import flax.linen as nn
import optax
import diffrax
import distrax
from sklearn.metrics import mean_absolute_error
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
import cv2
from jax.flatten_util import ravel_pytree
from data.dataloader_kth import load_data


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

    def __call__(self, params, t, x, y, args, pred=False):

        context = args['context']
        h = jax.vmap(jnp.interp, (None, None, 1))(t, context['ts'], context['hs'])

        if pred:
            print("PREDICTION\n")
            output = self.mlp.apply(params, jnp.concatenate([x, y.flatten(), h], axis=-1))
            return jnp.where(
                t > context['ts'][-1],
                jnp.zeros(self.num_latents),  # no control after context -> prior
                output,
            )
        else:
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

def count_params(params):
    flat_params, _ = ravel_pytree(params)
    return flat_params.size

def build_data_and_model(
        dataset_name: str,
        batch_size: int,
        white: bool,
        num_latents: int,
        num_contents: int,
        num_features: int,
        num_k: int,
        gamma_max: float,
        int_sub_steps: int,
        model_type: str,
    ):

    if white:
        num_k = 1
        gamma = None
        hurst = - 1
    else:
        gamma = gamma_by_gamma_max(num_k, gamma_max)
        hurst = None

    data_train, data_val, dataset_kwargs = data.get(batch_size, data_name = dataset_name)   ## Mnist
    ##data_train, data_val, dataset_kwargs = load_data(batch_size)                          ## KTH


    ##print(next(iter(data_train)))
    ##print(next(iter(data_train))[0].shape)
    ##print("AA: ", next(iter(data_train))[0].shape)
    ##print("BB: ", next(iter(data_train))[1].shape)
    if dataset_kwargs['image_size'] == 128:
        print("KTH MODE")
        num_contents=64*4
    dataset_kwargs['num_channels'] = next(iter(data_train))[0].shape[2]
    ts = jnp.arange(next(iter(data_train))[0].shape[1]+next(iter(data_train))[1].shape[1]) * dataset_kwargs['dt']
    dt = dataset_kwargs['dt'] / int_sub_steps

    key = jax.random.PRNGKey(0)
    b = Drift(num_latents)
    u = ControlFunction(num_k, num_latents, num_features)
    s = Diffusion(num_latents)
    sde = FractionalSDE(b, u, s, gamma, hurst=hurst, type=1, time_horizon=ts[-1], num_latents=num_latents)
    x0_prior = distrax.MultivariateNormalDiag(jnp.zeros(num_latents), jnp.ones(num_latents))
    model = VideoSDE(dataset_kwargs['image_size'], dataset_kwargs['num_channels'], num_features, num_latents, num_contents, x0_prior, True, sde, model_type)
    model._sde.check_dt(dt)
    params = model.init(key)
    ##param_count = sum(x.itemsize for x in jax.tree_leaves(params))
    ##print("\nPARAM COUNT: ", param_count, "\n")
    tot = 0
    for k in params.keys():
        kk = params[k]
        nk = count_params(kk)
        print(k, ": \t", nk)
        tot+=nk
    print(f"\n\nTotal Number of parameters: {tot}")

    return ts, dt, data_train, data_val, model, params


def train(
        dataset: str,
        white: bool = False,    # fallback to standard sde
        batch_size: int = 32,
        num_epochs: int = 200,
        num_latents: int = 4,
        num_contents: int = 64,
        num_features: int = 64,
        num_k: int = 5,
        gamma_max: float = 20.,
        int_sub_steps: int = 3,
        kl_weight: float = 1.,
        log_video_interval: int = 1000,
        model_type: str = "original",
    ):
    print("\n\n\n MODEL TYPE: ", model_type, "\n\n\n")
    solver = diffrax.StratonovichMilstein()

    ts, dt, data_train, data_val, model, params = build_data_and_model(dataset, batch_size, white, num_latents, num_contents, num_features, num_k, gamma_max, int_sub_steps, model_type)
    #dataloader = NumpyLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    dataloader=data_train
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

    @jax.jit
    def eval_fn(params, key, frames):
        frames_, _ = model(params, key, ts, frames, dt, solver)
        mse_loss = jnp.mean((frames - frames_) ** 2)
        mae_loss = jnp.mean(jnp.abs(frames - frames_))
        psnr = -10 * jnp.log10(mse_loss)
        return mse_loss*64*64, mae_loss*64*64, psnr


    @jax.jit
    def pred(params, key, frames, output):
        print("input shape: ", frames.shape)
        frames_, _ = model(params, key, ts, frames, dt, solver, pred=True)
        #print("F1 shape: ", frames_.shape)
        frames_ = frames_[10:,:,:,:]
        #print("AFTER: ", frames_.shape)
        #print("output: ", output.shape)
        mse_loss = jnp.mean((output - frames_) ** 2)
        mae_loss = jnp.mean(jnp.abs(output - frames_))
        psnr = -10 * jnp.log10(mse_loss)
        ##loss = ((frames_ - output)**2).sum()
        return mse_loss*64*64, mae_loss*64*64, psnr
    
    @jax.jit
    def pred_fn(params, key, frames, output, num_trials=3):
        trial_mse_losses = []
        trial_mae_losses = []
        trial_psnrs = []
        for _ in range(num_trials):
            random_key, subkey = jax.random.split(key)
            mse_loss, mae_loss, psnr = pred(params, subkey, frames, output)
            trial_mse_losses.append(mse_loss)
            trial_mae_losses.append(mae_loss)
            trial_psnrs.append(psnr)
        trial_mse_losses = jnp.array(trial_mse_losses)
        trial_mae_losses = jnp.array(trial_mae_losses)
        trial_psnrs = jnp.array(trial_psnrs)
        best_trial_idx = jnp.argmin(trial_mse_losses)
        return trial_mse_losses[best_trial_idx], trial_mae_losses[best_trial_idx], trial_psnrs[best_trial_idx]
    

    def batched_pred_fn(params, key, frames, output, batch_size=batch_size):
        keys = jax.random.split(key, batch_size)
        loss, mse_loss, mae_loss, psnr = jax.vmap(pred_fn, (None, 0, 0, 0))(params, keys, frames, output)
        return loss.mean()
    loss_grade = jax.jit(jax.value_and_grad(batched_pred_fn))


    
#     partition_optimizers = {'trainable': optax.adam(3e-4), 'frozen': optax.set_to_zero()}
#     param_partitions = traverse_util.path_aware_map(lambda path, v: 'frozen' if 'taesd' in path else 'trainable', params)
#     tx = optax.multi_transform(partition_optimizers, param_partitions)
    
#     print(param_partitions)

    optimizer = optax.adam(3e-4)
    opt_state = optimizer.init(params)
    random_key = jax.random.PRNGKey(7)
    for epoch in range(num_epochs):
        pbar = tqdm(range(len(dataloader)))
        vbar = tqdm(range(len(data_val)))
        tbar = tqdm(range(len(data_val)))

        
        for step, frames in zip(pbar, dataloader):

            frames = jnp.concatenate([frames[0].numpy(force=True).astype(onp.float32), frames[1].numpy(force=True).astype(onp.float32)], axis=1)
            frames = jnp.transpose(frames, (0, 1, 3, 4, 2))

            #input = frames[:,:10,:,:,:]
            #output = frames[:,10:,:,:,:]
            random_key, key = jax.random.split(random_key)
            (loss, loss_aux), grads = loss_grad(params, key, frames)
            nll, kl_x0, logpath = loss_aux
            #nll = loss.mean()
            #kl_x0, logpath = 0, 0
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            random_key, key = jax.random.split(random_key)
            keys = jax.random.split(key, batch_size)
            #mse_loss, mae_loss, psnr = jax.vmap(pred, (None, 0, 0, 0))(params, keys, input, output)

            pbar.set_description(f'[Epoch {epoch+1}/{num_epochs}] Loss: {float(loss):.2f}, Hurst: {model._sde.hurst(params["sde"]):.2f}, NLL: {nll:.2f}, KL_x0: {kl_x0:.2f}, KL_path: {logpath:.2f}')

            if onp.isnan(float(loss)):
                return

            wandb.log({
                'loss': float(loss),
                'nll': float(nll),
                'kl_x0': float(kl_x0),
                'kl_path': float(logpath),
                'hurst': float(model._sde.hurst(params["sde"]))
                #'train-pred-mse_loss': float(mse_loss.mean()),
                #'train-pred-mae_loss': float(mae_loss.mean()),
                #'train-pred-psnr': float(psnr.mean())
            })
            
            #if step % log_video_interval == 0:
             #   log_gif(model, frames, params, key, ts, dt, solver, (0, 1), "video_gif")

        p_mse_loss, p_mae_loss, p_psnr = 0, 0, 0
        v_mse_loss, v_mae_loss, v_psnr = 0, 0, 0
        for step_s, frames in zip(tbar, data_val):
            output = frames[1].numpy(force=True).astype(onp.float32)
            input = frames[0].numpy(force=True).astype(onp.float32)
            output = jnp.transpose(output, (0, 1, 3, 4, 2))
            input = jnp.transpose(input, (0, 1, 3, 4, 2))
            frames = jnp.concatenate([frames[0].numpy(force=True).astype(onp.float32), frames[1].numpy(force=True).astype(onp.float32)], axis=1)
            frames = jnp.transpose(frames, (0, 1, 3, 4, 2))
            random_key, key = jax.random.split(random_key)
            keys = jax.random.split(key, batch_size)
            mse_loss, mae_loss, psnr = jax.vmap(pred_fn, (None, 0, 0, 0))(params, keys, input, output)
            p_mse_loss += mse_loss.mean()
            p_mae_loss += mae_loss.mean()
            p_psnr += psnr.mean()

            mse_loss, mae_loss, psnr = jax.vmap(eval_fn, (None, 0, 0))(params, keys, frames)

            v_mse_loss += mse_loss.mean()
            v_mae_loss += mae_loss.mean()
            v_psnr += psnr.mean()
            tbar.set_description(f'[Epoch {epoch+1}/{num_epochs}] MSE: {float(mse_loss.mean()):.2f}, MAE: {float(mae_loss.mean()):.2f}, PSNR: {float(psnr.mean()):.2f}')
        
            if step_s%50==0:                      
                log_gif(model, frames, params, key, ts, dt, solver, (0, 1), "video_gif")
                pred_log_gif(model, frames, params, key, ts, dt, solver, (0, 1), "pred_video_gif")
        
        p_mse_loss = float(p_mse_loss) / step_s
        p_mae_loss = float(p_mae_loss) / step_s
        p_psnr = float(p_psnr) / step_s
        v_mse_loss = float(v_mse_loss) / step_s
        v_mae_loss = float(v_mae_loss) / step_s
        v_psnr = float(v_psnr) / step_s



        wandb.log({
            'pred-mse_loss': float(p_mse_loss),
            'pred-mae_loss': float(p_mae_loss),
            'pred-psnr': float(p_psnr),
            'mse_loss': float(v_mse_loss),
            'mae_loss': float(v_mae_loss),
            'psnr': float(v_psnr)
        })




      

        with open('params.p', 'wb') as f:
            pickle.dump(params, f)
        wandb.save('params.p')

    wandb.join(quiet=True)


'''
def log_gif(model, frames, params, key, ts, dt, solver, range, name):
    video, _ = model(params, key, ts, frames[0], dt, solver)
    video = jnp.repeat(video, 3, axis=-1)
    filename = f"tmp/{str(uuid.uuid4())}.gif"
    with imageio.get_writer(filename, mode="I") as writer:
        for b_frame in video:
            frame = (b_frame * 255).astype(jnp.uint8)
            writer.append_data(frame)

    wandb.log({
        name: wandb.Video(filename, fps=2, format="gif")
    })
    os.remove(filename)
'''

def log_gif(model, frames, params, key, ts, dt, solver, range, name):
    frames = jax.lax.stop_gradient(frames)
    video, _ = model(params, key, ts, frames[0], dt, solver)
    filename = f"tmp/{str(uuid.uuid4())}.gif"
    filename_recon = f"tmp/{str(uuid.uuid4())}_recon.gif"
    with imageio.get_writer(filename, mode="I") as writer:
        original_video = frames[0]
        if original_video.shape[-1]==1:
            original_video = jnp.tile(original_video, (1, 1, 1, 3))
        for b_frame in original_video:
            frame = (b_frame * 255).astype(jnp.uint8)
            writer.append_data(frame)
    with imageio.get_writer(filename_recon, mode="I") as writer:
        if video.shape[-1]==1:
            video = jnp.repeat(video, 3, axis=-1)
        for b_frame in video:
            frame = (b_frame * 255).astype(jnp.uint8)
            writer.append_data(frame)

    wandb.log({
        name: wandb.Video(filename, fps=2, format="gif"),
        name+"_recon": wandb.Video(filename_recon, fps=2, format="gif")
    })
    os.remove(filename)
    os.remove(filename_recon)
    
def pred_log_gif(model, frames, params, key, ts, dt, solver, rang, name, num_trials=3):
    frames = jax.lax.stop_gradient(frames)
    original_video = frames[0]
    frames = original_video[:10,:,:,:]
    
    best_video = None
    best_mse_loss = jnp.inf
    print(num_trials)
    for _ in range(num_trials):
        random_key, subkey = jax.random.split(key)
        video, _ = model(params, subkey, ts, frames, dt, solver, pred=True)
        mse_loss = jnp.mean((original_video[10:,:,:,:] - video[10:,:,:,:]) ** 2)
        
        if mse_loss < best_mse_loss:
            best_mse_loss = mse_loss
            best_video = video
    
    filename_recon = f"tmp/{str(uuid.uuid4())}_recon.gif"
    with imageio.get_writer(filename_recon, mode="I") as writer:
        if best_video.shape[-1] == 1:
            best_video = jnp.repeat(best_video, 3, axis=-1)
        for b_frame in best_video:
            frame = (b_frame * 255).astype(jnp.uint8)
            writer.append_data(frame)
    
    wandb.log({
        name+"_recon": wandb.Video(filename_recon, fps=2, format="gif")
    })
    os.remove(filename_recon)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_function_arguments(train, as_positional=False)

    cfg = parser.parse_args()
    wandb.init(project=f'jax-newest-{cfg.dataset}', config=cfg)
    train(**cfg)







'''
    def PSNR(pred, true, min_max_norm=True):
        """Peak Signal-to-Noise Ratio.

        Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
        """
        mse = jnp.mean((pred.astype(onp.float32) - true.astype(onp.float32))**2)
        print("DTYPE MSE: ", type(mse))
        if not True:
            return float('inf')
        else:
            if min_max_norm:  # [0, 1] normalized by min and max
                return 20. * jnp.log10(1. / jnp.sqrt(mse.mean()))  # i.e., -10. * np.log10(mse)
            else:
                return 20. * jnp.log10(255. / jnp.sqrt(mse.mean()))  # [-1, 1] normalized by mean and std
            
    def SSIM(pred, true, **kwargs):
    
        img1 = pred
        img2 = true
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2

        kernel = cv2.getGaussianKernel(11, 1.5)
        window = onp.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

'''
