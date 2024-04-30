import jax.numpy as jnp
import jax
import flax.linen as nn
import math

from einops import rearrange

class MultiTimeAttention(nn.Module):
    input_dim : int
    nhidden : int = 16
    embed_time : int = 16
    num_heads : int = 1

    def setup(self):
        assert self.embed_time % self.num_heads == 0
        self.embed_time_k = self.embed_time // self.num_heads

    def attention(self, query, key, value, mask, dropout):
        # TODO: Handle case where mask is not None
        dim = value.shape[-1]
        d_k = query.shape[-1]

        scores = jnp.matmul(query, jnp.transpose(key, (0, 1, 3, 2))) / math.sqrt(d_k)
        scores = jnp.repeat(jnp.expand_dims(scores, -1), dim, -1)
        p_attn = nn.softmax(scores, -2)
        return jnp.sum(p_attn * jnp.expand_dims(value, -3), -2), p_attn

    @nn.compact
    def __call__(self, query, key, value, mask = None, dropout = None):
        batch, _, dim = value.shape

        # TODO: Handle case where mask is not None

        value = jnp.expand_dims(value, 1)
        q = nn.Dense(self.embed_time)(query)
        k = nn.Dense(self.embed_time)(key)

        q = jnp.reshape(q, (q.shape[0], -1, self.num_heads, self.embed_time_k))
        q = jnp.transpose(q, (0, 2, 1, 3))

        k = jnp.reshape(k, (k.shape[0], -1, self.num_heads, self.embed_time_k))
        k = jnp.transpose(k, (0, 2, 1 ,3))

        x, _ = self.attention(q, k, value, mask, dropout)

        x = jnp.reshape(jnp.transpose(x, (0, 2, 1, 3)), (batch, -1, self.num_heads * dim))
        return nn.Dense(self.nhidden)(x)

class EncMtanRnn(nn.Module):
    input_dim : int
    query : jax.Array
    latent_dim : int = 2
    nhidden : int = 16
    embed_time : int = 16
    num_heads : int = 1
    learn_emb : bool = False

    def setup(self):
        if self.learn_emb:
            self.periodic = nn.Dense(self.embed_time - 1)
            self.linear = nn.Dense(1)

    @nn.compact
    def __call__(self, x, time_steps):
        time_steps = time_steps
        mask = x[:, :, self.input_dim:]
        if self.learn_emb:
            key = self.learn_time_embedding(time_steps)
            query = self.learn_time_embedding(jnp.expand_dims(self.query, 0))
        else:
            key = self.fixed_time_embedding(time_steps)
            query = self.fixed_time_embedding(jnp.expand_dims(self.query, 0))
        
        out = MultiTimeAttention(2 * self.input_dim, self.nhidden, self.embed_time, self.num_heads)(query, key, x, mask)
        out, _ = nn.Bidirectional(nn.RNN(nn.GRUCell(self.nhidden)), nn.RNN(nn.GRUCell(self.nhidden)))(out)
        out = nn.Dense(50)(out)
        out = nn.relu(out)
        out = nn.Dense(self.latent_dim * 2)
        return out
    
    def learn_time_embedding(self, tt):
        tt = jnp.expand_dims(tt, -1)
        out2 = jnp.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return jnp.concatenate([out1, out2], -1)
    
    def fixed_time_embedding(self, pos):
        d_model = self.embed_time
        pe = jnp.zeros((pos.shape[0], pos.shape[1], d_model))
        position = 48.0 * jnp.expand_dims(pos, 2)
        div_term = jnp.exp(jnp.arange(0, d_model, 2) * -(jnp.log(10.0) / d_model))
        pe[:, :, 0::2] = jnp.sin(position * div_term)
        pe[:, :, 1::2] = jnp.cos(position * div_term)
        return pe
        
class MTANEncoder(nn.Module):
    input_dim: int
    query: jax.Array
    nhidden: int = 16
    embed_time: int = 16
    num_heads: int = 1
    learn_emb: bool = False

    def learn_time_embedding(self, tt):
        tt = jnp.expand_dims(tt, -1)
        out2 = jnp.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return jnp.concatenate([out1, out2], -1)
    
    def fixed_time_embedding(self, pos):
#         print(f"Pos shape: {pos.shape}")
        d_model = self.embed_time
        pe = jnp.zeros((pos.shape[0], pos.shape[1], d_model))
        position = 48.0 * jnp.expand_dims(pos, 2)
        div_term = jnp.exp(jnp.arange(0, d_model, 2) * -(jnp.log(10.0) / d_model))
        pe.at[:, :, 0::2].set(jnp.sin(position * div_term))
        pe.at[:, :, 1::2].set(jnp.cos(position * div_term))
        return pe

    @nn.compact
    def __call__(self, x, time_steps):
        if len(x.shape) == 2:
            x = jnp.expand_dims(x, 0)
        mask = x[:, :, self.input_dim:]
        mask = jnp.concatenate([mask, mask], 2)
        if len(time_steps.shape) == 1:
            time_steps = jnp.expand_dims(time_steps, 0)
        if self.learn_emb:
            key = self.learn_time_embedding(time_steps)
            query = self.learn_time_embedding(jnp.expand_dims(self.query, 0))
        else:
            key = self.fixed_time_embedding(time_steps)
            query = self.fixed_time_embedding(jnp.expand_dims(self.query, 0))
        
        out = MultiTimeAttention(2 * self.input_dim, self.nhidden, self.embed_time, self.num_heads)(query, key, x, mask)
        out = nn.RNN(nn.GRUCell(self.nhidden))(out)
        return jnp.squeeze(out, 0)

class RecogNetwork(nn.Module):
    n_filters: int = 8

    def setup(self):
        self.map = nn.Sequential([
            nn.Conv(self.n_filters, kernel_size=(5, 5), strides=2, padding=(2, 2)),
            nn.BatchNorm(use_running_average=True),
            nn.relu,
            nn.Conv(self.n_filters * 2, kernel_size=(5, 5), strides=2, padding=(2, 2)),
            nn.BatchNorm(use_running_average=True),
            nn.relu,
            nn.Conv(self.n_filters * 4, kernel_size=(5, 5), strides=2, padding=(2, 2)),
            nn.BatchNorm(use_running_average=True),
            nn.relu,
            nn.Conv(self.n_filters * 8, kernel_size=(5, 5), strides=2, padding=(2, 2)),
            nn.BatchNorm(use_running_average=True),
            nn.relu,
        ])

    def __call__(self, x):
#         print(f"Start of recog network x shape: {x.shape}")
        h = self.map(x)
        h = jnp.reshape(h, (h.shape[0], -1))
        return h
    
class ReconNetwork(nn.Module):
    z_dim: int
    n_filters: int = 16

    def setup(self):
        self.initial_proj = nn.Dense(3 * 3 * 8)
        self.map = nn.Sequential([
            nn.ConvTranspose(self.n_filters * 8, kernel_size=(3, 3), strides=(2, 2), padding=(0, 0)),
            nn.relu,
            nn.ConvTranspose(self.n_filters * 4, kernel_size=(5, 5), strides=(2, 2), padding=(2, 2)), # TODO: Fix output_padding
            nn.relu,
            nn.ConvTranspose(self.n_filters * 2, kernel_size=(5, 5), strides=(2, 2), padding=(2, 2)), #TODO: Fix output_padding
            nn.relu,
            nn.ConvTranspose(1, kernel_size=(5, 5), strides=(1, 1), padding=(2, 2))
        ])
    
    def __call__(self, x):
        x = self.initial_proj(x)
        x = rearrange(x, "b (c h w) -> b h w c", b=x.shape[0], c=8, h=3, w=3)
        x = self.map(x)
        return x