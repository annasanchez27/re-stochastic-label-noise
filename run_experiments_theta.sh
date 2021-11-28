#!/bin/bash
python main.py --noise_rate 0.4 --dataset "cifar10" --noise_mode "symmetric" --use_sln True --perform_theta_experiment True --sigma 0.25
python main.py --noise_rate 0.4 --dataset "cifar10" --noise_mode "asymmetric" --use_sln True --perform_theta_experiment True --sigma 0.25
python main.py --noise_rate 0.4 --dataset "cifar10" --noise_mode "instance_dependent" --use_sln True --perform_theta_experiment True --sigma 0.25
python main.py --noise_rate 0.4 --dataset "cifar10" --noise_mode "openset" --use_sln True --perform_theta_experiment True --sigma 0.25

python main.py --noise_rate 0.4 --dataset "cifar10" --noise_mode "symmetric" --use_sln True --perform_theta_experiment True --sigma 0.5

python main.py --noise_rate 0.4 --dataset "cifar10" --noise_mode "symmetric" --use_sln True --perform_theta_experiment True --sigma 0.75
python main.py --noise_rate 0.4 --dataset "cifar10" --noise_mode "asymmetric" --use_sln True --perform_theta_experiment True --sigma 0.75
python main.py --noise_rate 0.4 --dataset "cifar10" --noise_mode "instance_dependent" --use_sln True --perform_theta_experiment True --sigma 0.75
python main.py --noise_rate 0.4 --dataset "cifar10" --noise_mode "openset" --use_sln True --perform_theta_experiment True --sigma 0.75

python main.py --noise_rate 0.4 --dataset "cifar10" --noise_mode "asymmetric" --use_sln True --perform_theta_experiment True --sigma 1.0
python main.py --noise_rate 0.4 --dataset "cifar10" --noise_mode "instance_dependent" --use_sln True --perform_theta_experiment True --sigma 1.0
python main.py --noise_rate 0.4 --dataset "cifar10" --noise_mode "openset" --use_sln True --perform_theta_experiment True --sigma 1.0
