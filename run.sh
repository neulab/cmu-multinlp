#!/usr/bin/env bash
#SBATCH --mem=30000
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --output=slurm_out/slurm-%j.out

config=$1
output=$2
args="${@:3}"

allennlp train ${config} -s ${output} --include-package brat_multitask ${args}
