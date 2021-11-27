#!/bin/bash
for use_sln in True False; do
  for dataset in "cifar10"; do  # "cifar100"
    for noise_mode in "symmetric" "asymmetric" "instance_dependent" "openset"; do
        python main.py --noise_rate 0.4 --dataset "${dataset}" --noise_mode "${noise_mode}" --use_sln "${use_sln}"
    done
  done
done
