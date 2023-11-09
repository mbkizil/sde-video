In your virtual environment add these modules from KUACC cluster:

```
module load cuda/11.8.0   
module load cudnn/8.6.0/cuda-11.x
```
Install jax using this line:

```
pip install --upgrade "jax[cuda11_pip]==0.4.19" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Then follow the steps in the readme file of VideoAsAStochasticProcess.
