from setuptools import setup, find_packages

setup(
    name="video-sde",
    version="0.0.2",
    install_requires=[
        'jax==0.4.23',
        'scipy==1.11.3',
        'numpy==1.26.1',
        'notebook==6.5.6',
        'matplotlib==3.8.0',
        'wandb==0.15.12',
        'moviepy==1.0.3',
        'imageio==2.31.6',
        'jsonargparse==4.26.1',
        'flax==0.7.4',
        'optax==0.1.7',
        'diffrax==0.4.1',
        'distrax @ git+https://github.com/google-deepmind/distrax@a6b19eccea1abe69874483641371786d4ed44d6e',
        'pandas==2.1.1',
        'seaborn==0.13.0',
        'tqdm==4.66.1',
    ],
    packages=find_packages(include=["sde"]),
)