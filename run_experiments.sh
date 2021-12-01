#!/bin/bash
for use_sln in false true; do
  for noise_mode in "openset" "symmetric" "asymmetric" "instance_dependent"; do
    if [[ $use_sln =  true ]];
      then
          python main.py --noise_rate 0.4 --dataset "cifar10" --noise_mode "${noise_mode}" --use_sln "${use_sln}"
      else
          python main.py --noise_rate 0.4 --dataset "cifar10" --noise_mode "${noise_mode}"
      fi
  done
done

for use_sln in false true; do
  for noise_mode in "openset" "symmetric" "asymmetric" "instance_dependent"; do
    if [[ $use_sln =  true ]];
      then
          python main.py --noise_rate 0.4 --dataset "cifar100" --noise_mode "${noise_mode}" --use_sln "${use_sln}"
      else
          python main.py --noise_rate 0.4 --dataset "cifar100" --noise_mode "${noise_mode}"
      fi
  done
done
