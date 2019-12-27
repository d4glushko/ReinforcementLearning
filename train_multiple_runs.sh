#!/bin/bash

EXECUTIONS_NUMBER=${1:-1}

AGENTS_NUMBER=2
ENV_NAME='CartPole-v1'
AGENT='A2C'
NOISE_ENV_STEP=0.1
ENABLE_EXCHANGE=true

WARM_UP_STEPS=30
EXCHANGE_STEPS=5

DEBUG=false
USE_CUDA=false
TRAINING_EPISODES=500

SECONDS=0

for ((i=1; i <= $EXECUTIONS_NUMBER; i++))
do
    python3 diploma/main.py --current_execution=$i --total_executions=$EXECUTIONS_NUMBER --agents_number=$AGENTS_NUMBER --env_name=$ENV_NAME --agent=$AGENT --noise_env_step=$NOISE_ENV_STEP --debug=$DEBUG --use_cuda=$USE_CUDA --training_episodes=$TRAINING_EPISODES --enable_exchange=$ENABLE_EXCHANGE --warm_up_steps=$WARM_UP_STEPS --exchange_steps=$EXCHANGE_STEPS
done

DURATION=$SECONDS
echo "Total execution time: $(($DURATION / 60)) minutes and $(($DURATION % 60)) seconds."
