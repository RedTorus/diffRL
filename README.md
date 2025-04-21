
## Requirements
Installations of [PyTorch](https://pytorch.org/) and [MuJoCo](https://github.com/deepmind/mujoco) are needed. 
A suitable [conda](https://conda.io) environment named `qvpo` can be created and activated with:
```
conda create qvpo
conda activate qvpo
```
To get started, install the additionally required python packages into you environment.
```
pip install -r requirements.txt
```

## Running
Running experiments based our code could be quite easy, so below we use `HalfCheetah-v3` task as an example. 

```
python main.py --env_name HalfCheetah-v3--weighted --aug
```



## Acknowledgement

The code  is based on the implementation of [DIPO](https://github.com/BellmanTimeHut/DIPO), [QVPO](https://github.com/wadx2019/qvpo)
=======
# diffRL
