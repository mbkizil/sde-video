from typing import Any, Callable
import jax
from jax import random, numpy as jnp, image
import flax
from flax import linen as nn
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF

import jax.numpy as jnp
import numpy as np
import torch

def conv(n_in, n_out, kernel_size=3, stride=1, **kwargs):
    return nn.Conv(features=n_out, kernel_size=(kernel_size, kernel_size), strides=(stride, stride),
                   padding=1, **kwargs)

class ReLU(nn.Module):
    def __call__(self, x):
        return nn.relu(x)

# class Upsample(nn.Module):
#     def __call__(self, x):
#         return jax.image.resize(x, (x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3]), method='bilinear')
    
class Upsample(nn.Module):
    def __call__(self, x):
        B, H, W, C = x.shape

        return jax.image.resize(x, shape=(B, H*2, W*2, C), method="nearest")

@nn.compact
class Clamp(nn.Module):
    
    def __call__(self, x):
        return jnp.tanh(x / 3) * 3

class Block(nn.Module):
    n_out: int

    @nn.compact
    def __call__(self, x):
        n_in = x.shape[-1]
        
        residual = x
        # Convolution 1
        x = conv(n_in, self.n_out)(x)
        x = nn.relu(x)
        
        # Convolution 2
        x = conv(self.n_out, self.n_out)(x)
        x = nn.relu(x)
        
        # Convolution 3 without downsampling
        x = conv(self.n_out, self.n_out)(x)

        x = x + residual
        x = nn.relu(x)
        return x

class Encoder_taesd(nn.Module):

    @nn.compact
    def __call__(self, x):
        x = conv(3, 64)(x)
        x = Block(n_out=64)(x)


        x = conv(64, 64, stride=2, use_bias=False)(x)
        x = Block(n_out=64)(x)
        x = Block(n_out=64)(x)
        x = Block(n_out=64)(x)

        x = conv(64, 64, stride=2, use_bias=False)(x)
        x = Block(n_out=64)(x)
        x = Block(n_out=64)(x)
        x = Block(n_out=64)(x)

        x = conv(64, 64, stride=2, use_bias=False)(x)
        x = Block(n_out=64)(x)
        x = Block(n_out=64)(x)
        x = Block(n_out=64)(x)

        x = conv(64, 4, use_bias=False)(x)

        return x

class Decoder_taesd(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = Clamp()(x)
        x = conv(4, 64)(x)
        x = ReLU()(x)

        x = Block(64)(x)
        x = Block(64)(x)
        x = Block(64)(x)

        x = Upsample()(x)
        x = conv(x, 64, use_bias=False)(x)

        x = Block(64)(x)
        x = Block(64)(x)
        x = Block(64)(x)

        x = Upsample()(x)
        x = conv(x, 64, use_bias=False)(x)

        x = Block(64)(x)
        x = Block(64)(x)
        x = Block(64)(x)

        x = Upsample()(x)
        x = conv(x, 64, use_bias=False)(x)

        x = Block(64)(x)
        x = conv(x, 3, use_bias=True)(x)
        return x

class TAESD(nn.Module):

    def __init__(self):
        self.latent_magnitude = 3
        self.latent_shift = 0.5
        self.encoder = Encoder_taesd()
        self.decoder = Decoder_taesd()

        
    def scale_latents(self, x):
        temp = x / (2 * self.latent_magnitude) + self.latent_shift
        print(jnp.min(temp), jnp.max(temp))
        return jnp.clip(temp, 0, 1)

    def unscale_latents(self, x):
        return (x - self.latent_shift) * (2 * self.latent_magnitude)


    def init(self, rng, image):
        # Initialize parameters
        params = {}
        rng, init_rng = jax.random.split(rng)
        encoder_params = self.encoder.init(init_rng, image)
        decoder_params = self.decoder.init(init_rng, jnp.zeros((1, 1, 100, 4)))
        params['encoder'] = encoder_params
        params['decoder'] = decoder_params

        return params

    def apply_encoder(self, params, *args):
        return self.encoder.apply(params['encoder'], *args)

    def apply_decoder(self, params, *args):
        return self.decoder.apply(params['decoder'], *args)

    def __call__(self, params, image):
        print(f"Original min and max: {jnp.min(image)}, {jnp.max(image)}")
        encoded_image = self.apply_encoder(params, image)
        scaled_encoded_image = self.scale_latents(encoded_image)
        
        # Save encoded image
        encoded_path = "test_image_encoded.png"
        scaled_encoded_np = np.array(scaled_encoded_image[0] * 255).astype(np.uint8)
        print(f"After encoder and everything: {np.min(scaled_encoded_np)}, {np.max(scaled_encoded_np)}")
        Image.fromarray(scaled_encoded_np).save(encoded_path)
        
        # Unscaling image
        encoded_unscaled = self.unscale_latents(scaled_encoded_image)
        print(f"After unscaling: {jnp.min(encoded_unscaled)}, {jnp.max(encoded_unscaled)}")
        decoded_image = self.apply_decoder(params, encoded_unscaled)
        print(f"After decoder: {jnp.min(decoded_image)}, {jnp.max(decoded_image)}")
        decoded_image = jnp.clip(decoded_image, 0, 1)
        print(f"After clamp: {jnp.min(decoded_image)}, {jnp.max(decoded_image)}")
        # decoded_image = np.array(jnp.clip(self.apply_decoder(params, self.unscale_latents(scaled_encoded_image)), 0, 1))
        
        # Save decoded image
        decoded_path = "test_image_decoded.png"
        scaled_decoded_np = np.array(decoded_image[0] * 255).astype(np.uint8)
        Image.fromarray(scaled_decoded_np).save(decoded_path)
        
        return decoded_image


def convert_pytorch_weights_to_jax(pytorch_state_dict):
    jax_params = {}

    for pt_name, pt_param in pytorch_state_dict.items():
        # Split the PyTorch parameter name into parts
        parts = pt_name.split('.')
        
        # Determine if this is a convolutional layer or a bias
        is_conv = 'weight' in parts[-1]  # Assumes weight names contain 'weight'
        is_bias = 'bias' in parts[-1]    # Assumes bias names contain 'bias'
        
        # Map PyTorch layer names to JAX/Flax structure
        if len(parts) == 2:  # Direct conv layer or bias in the root (e.g., "0.weight")
            block_name = f"Conv_{parts[0]}"  # Map to Conv_0, Conv_1, etc.
            param_type = 'kernel' if is_conv else 'bias'
        elif len(parts) > 2:  # Nested conv layers (e.g., "1.conv.0.weight")
            block_index = parts[0]
            conv_index = parts[2]
            block_name = f"Block_{block_index}"
            conv_name = f"Conv_{conv_index}"
            param_type = 'kernel' if is_conv else 'bias'
        
        # Prepare parameter for JAX/Flax
        param_value = np.array(pt_param.cpu().detach().numpy())
        if is_conv:
            # Transpose PyTorch conv weights (O, I, H, W) to JAX (H, W, I, O)
            param_value = param_value.transpose((2, 3, 1, 0))
        
        # Assign to the correct place in JAX parameters
        if len(parts) > 2:
            if block_name not in jax_params:
                jax_params[block_name] = {}
            if conv_name not in jax_params[block_name]:
                jax_params[block_name][conv_name] = {}
            jax_params[block_name][conv_name][param_type] = jnp.asarray(param_value)
        else:
            if block_name not in jax_params:
                jax_params[block_name] = {}
            jax_params[block_name][param_type] = jnp.asarray(param_value)
    
    return {'params': jax_params}


def get_params_for_layer(layer_num, state_dict):
    layer_params = {}

    for name, param in state_dict.items():
        name_parts = name.split(".")
        if name_parts[0] == layer_num:
            layer_params[name] = param

    return layer_params

def torch_to_jax_encoder(state_dict):
    layer_num_dict = {0: "Conv_0", 1: "Block_0", 2: "Conv_1", 3: "Block_1", 4: "Block_2",
                        5: "Block_3", 6: "Conv_2", 7: "Block_4", 8: "Block_5", 9: "Block_6",
                        10: "Conv_3", 11: "Block_7", 12: "Block_8", 13: "Block_9", 14: "Conv_4"}
    jax_state_dict = {"params": {}}

    for i in range(15):
        layer_state_dict = get_params_for_layer(str(i), state_dict)

        # jax_layer_dict = {"params": {}}
        jax_layer_dict = {}
        # This means that the layer is a single conv layer with bias
        if i == 0 or i == 14:
            kernel = layer_state_dict[f'{i}.weight'].cpu().numpy()
            bias = layer_state_dict[f'{i}.bias'].cpu().numpy()
            jax_layer_dict['kernel'] = jnp.transpose(kernel, (2, 3, 1, 0))
            jax_layer_dict['bias'] = bias

            if i == 0:
                jax_state_dict['params']['Conv_0'] = jax_layer_dict
            else:
                jax_state_dict['params']['Conv_4'] = jax_layer_dict
        # The layer is a single conv layer with no bias
        elif i == 2 or i == 6 or i == 10:
            kernel = layer_state_dict[f'{i}.weight'].cpu().numpy()

            jax_layer_dict['kernel'] = jnp.transpose(kernel, (2, 3, 1, 0))

            if i == 2:
                jax_state_dict['params']["Conv_1"] = jax_layer_dict
            elif i == 6:
                jax_state_dict['params']["Conv_2"] = jax_layer_dict
            else:
                jax_state_dict['params']["Conv_3"] = jax_layer_dict
        # Layer is a block layer. Block layers have 3 conv layers
        else:
            kernel1 = layer_state_dict[f"{i}.conv.0.weight"].cpu().numpy()
            bias1 = layer_state_dict[f"{i}.conv.0.bias"].cpu().numpy()
            kernel2 = layer_state_dict[f"{i}.conv.2.weight"].cpu().numpy()
            bias2 = layer_state_dict[f"{i}.conv.2.bias"].cpu().numpy()
            kernel3 = layer_state_dict[f"{i}.conv.4.weight"].cpu().numpy()
            bias3 = layer_state_dict[f"{i}.conv.4.bias"].cpu().numpy()

            kernel1 = jnp.transpose(kernel1, (2, 3, 1, 0))
            kernel2 = jnp.transpose(kernel2, (2, 3, 1, 0))
            kernel3 = jnp.transpose(kernel3, (2, 3, 1, 0))

            jax_layer_dict['Conv_0'] = {"kernel": kernel1, "bias": bias1}
            jax_layer_dict['Conv_1'] = {"kernel": kernel2, "bias": bias2}
            jax_layer_dict['Conv_2'] = {"kernel": kernel3, "bias": bias3}

            jax_state_dict['params'][layer_num_dict[i]] = jax_layer_dict
    # print(jax_state_dict.keys())
    return jax_state_dict

def torch_to_jax_decoder(state_dict):
    layer_num_dict = {1: "Conv_0", 3: "Block_0", 4: "Block_1", 5: "Block_2", 7: "Conv_1", 
                      8: "Block_3", 9: "Block_4", 10: "Block_5", 12: "Conv_2", 13: "Block_6",
                      14: "Block_7", 15: "Block_8", 17: "Conv_3", 18: "Block_9", 19: "Conv_4"}
    jax_state_dict = {"params": {}}

    for i in range(20):
        if i not in layer_num_dict.keys():
            continue
        layer_state_dict = get_params_for_layer(str(i), state_dict)

        # jax_layer_dict = {"params": {}}
        jax_layer_dict = {}
        # This means that the layer is a single conv layer with bias
        if i == 1 or i == 19:
            kernel = layer_state_dict[f'{i}.weight'].cpu().numpy()
            bias = layer_state_dict[f'{i}.bias'].cpu().numpy()
            jax_layer_dict['kernel'] = jnp.transpose(kernel, (2, 3, 1, 0))
            jax_layer_dict['bias'] = bias

            # if i == 0:
            #     jax_state_dict['params']['Conv_0'] = jax_layer_dict
            # else:
            #     jax_state_dict['params']['Conv_4'] = jax_layer_dict
            jax_state_dict['params'][layer_num_dict[i]] = jax_layer_dict
        # The layer is a single conv layer with no bias
        elif i == 7 or i == 12 or i == 17:
            kernel = layer_state_dict[f'{i}.weight'].cpu().numpy()

            jax_layer_dict['kernel'] = jnp.transpose(kernel, (2, 3, 1, 0))

            jax_state_dict['params'][layer_num_dict[i]] = jax_layer_dict
        # Layer is a block layer. Block layers have 3 conv layers
        else:
            kernel1 = layer_state_dict[f"{i}.conv.0.weight"].cpu().numpy()
            bias1 = layer_state_dict[f"{i}.conv.0.bias"].cpu().numpy()
            kernel2 = layer_state_dict[f"{i}.conv.2.weight"].cpu().numpy()
            bias2 = layer_state_dict[f"{i}.conv.2.bias"].cpu().numpy()
            kernel3 = layer_state_dict[f"{i}.conv.4.weight"].cpu().numpy()
            bias3 = layer_state_dict[f"{i}.conv.4.bias"].cpu().numpy()

            kernel1 = jnp.transpose(kernel1, (2, 3, 1, 0))
            kernel2 = jnp.transpose(kernel2, (2, 3, 1, 0))
            kernel3 = jnp.transpose(kernel3, (2, 3, 1, 0))

            jax_layer_dict['Conv_0'] = {"kernel": kernel1, "bias": bias1}
            jax_layer_dict['Conv_1'] = {"kernel": kernel2, "bias": bias2}
            jax_layer_dict['Conv_2'] = {"kernel": kernel3, "bias": bias3}

            jax_state_dict['params'][layer_num_dict[i]] = jax_layer_dict
    # print(jax_state_dict.keys())
    return jax_state_dict
