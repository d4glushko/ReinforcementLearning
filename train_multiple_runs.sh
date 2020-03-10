#!/bin/bash

EXECUTIONS_NUMBER=${1:-1}

AGENTS_NUMBER=1
ENV_NAME='Breakout-v4'

# Available agents: DQN, A2C
AGENT='DQN'
NOISE_ENV_STEP=0.

NOISE_DROPOUT_STEP=0.

# Whether to give smaller eps to envs with higher noise. Applicable only for DQN currently and only for initial state (doesn't take into account exchanges)
EPS_WRT_NOISE=false

# Available types: NO, RANDOM, SMART
EXCHANGE_TYPE='NO'
EXCHANGE_DELTA=0.1
EXCHANGE_ITEMS_REWARD_COUNT=30

WARM_UP_STEPS=30
EXCHANGE_STEPS=5
EARLY_STOPPING=false

DEBUG=false
USE_CUDA=false
TRAINING_EPISODES=4000

NUM_STEPS_PER_EPISODE=0 #specify number of steps if you want every agent to train for the same number of steps
PLAY_EPISODES=500
IGNORE_PLAY=false

DATE=$(date +%s)
SECONDS=0

for ((i=1; i <= $EXECUTIONS_NUMBER; i++))
do
    python3 diploma/main.py --date=$DATE --current_execution=$i --total_executions=$EXECUTIONS_NUMBER --agents_number=$AGENTS_NUMBER --env_name=$ENV_NAME --agent=$AGENT --noise_env_step=$NOISE_ENV_STEP --noise_dropout_step=$NOISE_DROPOUT_STEP --debug=$DEBUG --use_cuda=$USE_CUDA --training_episodes=$TRAINING_EPISODES --exchange_type=$EXCHANGE_TYPE --warm_up_steps=$WARM_UP_STEPS --exchange_steps=$EXCHANGE_STEPS --exchange_delta=$EXCHANGE_DELTA --exchange_items_reward_count=$EXCHANGE_ITEMS_REWARD_COUNT --play_episodes=$PLAY_EPISODES --ignore_play=$IGNORE_PLAY --epsilon_wrt_noise=$EPS_WRT_NOISE --early_stopping=$EARLY_STOPPING --num_steps_per_episode=$NUM_STEPS_PER_EPISODE
done

DURATION=$SECONDS
echo "Total execution time: $(($DURATION / 60)) minutes and $(($DURATION % 60)) seconds."
