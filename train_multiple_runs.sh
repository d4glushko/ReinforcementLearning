#!/bin/bash

EXECUTIONS_NUMBER=${1:-10}

AGENTS_NUMBER=10
ENV_NAME='CartPole-v1'
AGENT='DQN'
NOISE_ENV_STEP=0.1

METRICS_NUMBER_OF_ELEMENTS=100
METRICS_NUMBER_OF_ITERATIONS=50

DEBUG=false
USE_CUDA=true
TRAINING_EPISODES=1000


for ((i=1; i <= $EXECUTIONS_NUMBER; i++))
do
    python diploma/main.py --current_execution=$i --total_executions=$EXECUTIONS_NUMBER --agents_number=$AGENTS_NUMBER --env_name=$ENV_NAME --agent=$AGENT --noise_env_step=$NOISE_ENV_STEP --metrics_number_of_elements=$METRICS_NUMBER_OF_ELEMENTS --metrics_number_of_iterations=$METRICS_NUMBER_OF_ITERATIONS --debug=$DEBUG --use_cuda=$USE_CUDA --training_episodes=$TRAINING_EPISODES
done
