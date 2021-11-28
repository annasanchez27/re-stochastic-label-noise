#!/bin/bash
for dataset in "cifar10" "cifar100"; do
  for use_sln in false true false; do
    for noise_mode in "symmetric" "asymmetric" "instance_dependent" "openset"; do
      if [[ $use_sln =  true ]];
        then
            python main.py --noise_rate 0.4 --dataset "${dataset}" --noise_mode "${noise_mode}" --use_sln "${use_sln}"
        else
            python main.py --noise_rate 0.4 --dataset "${dataset}" --noise_mode "${noise_mode}"
        fi
    done
  done
done
