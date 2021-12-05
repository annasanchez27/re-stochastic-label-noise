#!/bin/bash
for noise_mode in "openset" "symmetric" "asymmetric" "instance_dependent"; do
      python main.py --noise_rate 0.4 --dataset "cifar10" --noise_mode "${noise_mode}" --use_sln true --trainable_variance true
done
