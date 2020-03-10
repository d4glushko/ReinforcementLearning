#!/bin/bash

AGENTS_NUMBER=10
ENV_NAME='CartPole-v1'

# Available agents: DQN, A2C
AGENT='A2C'
NOISE_ENV_STEP=0.1
NOISE_DROPOUT_STEP=0.
EARLY_STOPPING=false

# Available types: NO, RANDOM, SMART
EXCHANGE_TYPE='SMART'
EXCHANGE_DELTA=0.1
EXCHANGE_ITEMS_REWARD_COUNT=30
NUM_STEPS_PER_EPISODE=500

DETAILED_AGENTS_PLOTS=false
METRICS_NUMBER_OF_ELEMENTS=100
METRICS_NUMBER_OF_ITERATIONS=50

IGNORE_PLAY=false

#EXECUTIONS_COUNT=
# EXECUTIONS_FROM=0
EXECUTION_DATE='2020-03-08_20-07-47'

python diploma/show_results.py --execution_date=$EXECUTION_DATE --agents_number=$AGENTS_NUMBER --env_name=$ENV_NAME --agent=$AGENT --noise_env_step=$NOISE_ENV_STEP --noise_dropout_step=$NOISE_DROPOUT_STEP --metrics_number_of_elements=$METRICS_NUMBER_OF_ELEMENTS --metrics_number_of_iterations=$METRICS_NUMBER_OF_ITERATIONS --exchange_type=$EXCHANGE_TYPE --executions_count=$EXECUTIONS_COUNT --executions_from=$EXECUTIONS_FROM --exchange_delta=$EXCHANGE_DELTA --exchange_items_reward_count=$EXCHANGE_ITEMS_REWARD_COUNT --detailed_agents_plots=$DETAILED_AGENTS_PLOTS --ignore_play=$IGNORE_PLAY --early_stopping=$EARLY_STOPPING --num_steps_per_episode=$NUM_STEPS_PER_EPISODE
