#!/bin/bash
for noise_mode in "symmetric" "asymmetric" "instance_dependent" "openset"; do
    for sln_mode in "noisy" "clean";do
      python main.py --noise_rate 0.4 --dataset "cifar10" --noise_mode "${noise_mode}" --use_sln --sln_mode "${sln_mode}" --perform_sln_experiment
      done
done

