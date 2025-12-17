# Introduction
This repository has been created for the purposes of teaching Deep Learning classes
at Prague University of Economics and Business, Faculty of Informatics and Statistics. 
It has been created by Martin Kadlec (@R4tmax) during the course of AY 2024/2025 and is 
published without a license. 

Code demonstrates usage of CrewAI as LLM orchestration tool for RAG over Obsidian Vault.


## Setup 
See **config.py** and **requirements.txt**, note that reqs were build using pipreqes, I believe that
crewAI ecosystem has some unfortunate interactions with pip dependencies and as such file is not comprehensive.

I also provided a freeze output, so if you want, you can instead reconstruct my entire env on docker, conda or other
tooling of choice.

Code is tested on Python 3.10.15 but should be also runnable on most higher version instances. 

To launch locally you need
- LLM model instance, I recommend using API keys for foundational models as HF models have historically been tricky to get to work with CrewAI framework
- Create .env file as per config.py
- Local dir with Obsidian Vault (any collection of .md files in theory)

Launch with 
```cmd
 streamlit run main.py
```
from project root, other attempts will cause threading errors.

I use Conda on my machine, but for work and non-legacy projects I migrated to **uv**, poetry is also a good option 
as far as I can tell.

## Known Issues
On startup you can get any permutation of warnings

```
2025-12-16 20:55:26.428268: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-12-16 20:55:49.125462: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
WARNING:tensorflow:From C:\Users\kadle\anaconda3\envs\MLOps\lib\site-packages\tf_keras\src\losses.py:2976: The name tf.losses.sparse_softmax_cross_entropy is deprecated. Please use tf.compat.v1.losses.sparse_softmax_cross_entropy instead.
WARNING:tensorflow:From C:\Users\kadle\anaconda3\envs\MLOps\lib\site-packages\keras\src\backend\common\global_state.py:82: The name tf.reset_default_graph is deprecated. Please use tf.compat.v1.reset_default_graph instead.

2025-12-16 20:57:38.781 Examining the path of torch.classes raised: Tried to instantiate class '__path__._path', but it does not exist! Ensure that it is registered via torch::class_

```

This is due to packaging issues with lot of the CPU-only libraries in python and are not indicative of 
problem with environment.

If you have issues with Torch/Keras init its probably most likely due to you installing default/CUDA versions.
Most laptops need CPU only libraries otherwise interpreter goes ballistic.
