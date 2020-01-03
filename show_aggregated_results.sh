#!/bin/bash

AGENTS_NUMBER=10
ENV_NAME='CartPole-v1'

# Available agents: DQN, A2C
AGENT='A2C'
NOISE_ENV_STEP=0.1

# Available types: NO, RANDOM, SMART
EXCHANGE_TYPE='SMART'
EXCHANGE_DELTA=0.1
EXCHANGE_ITEMS_REWARD_COUNT=30

DETAILED_AGENTS_PLOTS=false
METRICS_NUMBER_OF_ELEMENTS=100
METRICS_NUMBER_OF_ITERATIONS=50

# EXECUTIONS_COUNT=1
# EXECUTIONS_FROM=0
# DATE='2020-01-03_10:22:52'


python diploma/show_results.py --date=$DATE --agents_number=$AGENTS_NUMBER --env_name=$ENV_NAME --agent=$AGENT --noise_env_step=$NOISE_ENV_STEP --metrics_number_of_elements=$METRICS_NUMBER_OF_ELEMENTS --metrics_number_of_iterations=$METRICS_NUMBER_OF_ITERATIONS --exchange_type=$EXCHANGE_TYPE --executions_count=$EXECUTIONS_COUNT --executions_from=$EXECUTIONS_FROM --exchange_delta=$EXCHANGE_DELTA --exchange_items_reward_count=$EXCHANGE_ITEMS_REWARD_COUNT --detailed_agents_plots=$DETAILED_AGENTS_PLOTS
