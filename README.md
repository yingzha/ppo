### PPO impementation using pytorch
The code was adapted from https://github.com/vwxyzjn/ppo-implementation-details.

Please refer to https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/ for more reading.


## Get started
1. build the docker images
```python
docker build . -t ppo-dev 
```
2. create a container and expose the port (for tensorboard)
```python
# specify OUPUT_DIR 
docker run -v $OUTPUT_DIR:/app/ -p 6006:6006 --rm -it ppo-dev /bin/bash
```
3. run the training script (with your own parameters) 
```python
python train.py num_iterations=50000 rollout_length=100
```

The model file, training logs and evaluation video can be found in $OUTPUT_DIR.
