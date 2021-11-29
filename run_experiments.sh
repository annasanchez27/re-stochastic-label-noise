#!/bin/bash
#for dataset in "cifar10" "cifar100"; do
#  for use_sln in false true; do
#    for noise_mode in "openset" "symmetric" "asymmetric" "instance_dependent"; do
#      if [[ $use_sln =  true ]];
#        then
#            python main.py --noise_rate 0.4 --dataset "${dataset}" --noise_mode "${noise_mode}" --use_sln "${use_sln}"
#        else
#            python main.py --noise_rate 0.4 --dataset "${dataset}" --noise_mode "${noise_mode}"
#        fi
#    done
#  done
#done

for dataset in "cifar100"; do
  for use_sln in false true; do
    for noise_mode in "openset" "symmetric" "asymmetric" "instance_dependent"; do
      if [[ $use_sln =  true ]];
        then
            python main.py --noise_rate 0.4 --dataset "${dataset}" --noise_mode "${noise_mode}" --use_sln "${use_sln}"
        else
            python main.py --noise_rate 0.4 --dataset "${dataset}" --noise_mode "${noise_mode}"
        fi
    done
  done
done
