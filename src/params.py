import os
import json
from utils import *
import argparse
def setup_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params')
    return parser.parse_args()
def default_params(params):
    def set_default(name, value):
        if name not in params:
            params[name] = value
    set_default('logs_path', 'logs/texturizeL0')
    set_default('lr', 1e-4)
    set_default('load_checkpoint', False)
    set_default('checkpoint', None)
    set_default('train_file', 'data/train.txt')
    set_default('test_file', 'data/test.txt')
    set_default('is_training', False)
    set_default('texture_queue', None)
    set_default('num_epochs', 180)
    set_default('resize_images', True)
    set_default('resize_size', (600, 800))
    set_default('initializer', 'identity')
    set_default('continue_checkpoint', None)
def read_params(params_file):
    import json
    with open(params_file) as f:
        params = json.load(f)
        return params
def write_params(params, directory):
    import json
    with open(pj(directory, 'params.txt'), 'w') as f:
        json.dump(params, f, indent=4, sort_keys=True)
