
## Requirements
Installations of [PyTorch](https://pytorch.org/) and [MuJoCo](https://github.com/deepmind/mujoco) are needed. 
A suitable [conda](https://conda.io) environment named `qvpo` can be created and activated with:
```
conda create diffRL
conda activate diffRL
```
To get started, install the additionally required python packages into you environment.
```
pip install -r requirements2.txt
```

## Running
`HalfCheetah-v3` task as an example. 

```
python main.py --env_name HalfCheetah-v3--weighted --aug
```
or more complex
```
python main.py --env_name relocate-human-v0 --agent dipo --n_timesteps 40 --batch_size 128 --pretraining_steps 10000
```


## Acknowledgement
The code  is based on the implementation of [DIPO](https://github.com/BellmanTimeHut/DIPO), [QVPO](https://github.com/wadx2019/qvpo)

=======
# diffRL
