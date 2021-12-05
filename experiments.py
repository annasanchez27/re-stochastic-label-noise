from pathlib import Path
import numpy as np
import yaml

from model.wrn import WideResNet

config = yaml.load(Path("config.yml").read_text(), Loader=yaml.SafeLoader)
mean = config['cifar10']['mean']
variance = np.square(config['cifar10']['std'])
model = WideResNet(mean, variance, 0, ga_steps=config["accumulation_steps"], inputs=(None, 32, 32, 3), sln_mode=None,
                   num_classes=10)
model.load_weights("checkpoints/crossentropy/final_model_cifar10_symmetric_0.4_0.0")
