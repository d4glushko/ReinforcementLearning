#!/bin/bash

AGENTS_NUMBER=10
ENV_NAME='CartPole-v1'
AGENT='DQN'
NOISE_ENV_STEP=0.1
ENABLE_EXCHANGE=true

METRICS_NUMBER_OF_ELEMENTS=100
METRICS_NUMBER_OF_ITERATIONS=50

# EXECUTIONS_COUNT = 1
# EXECUTIONS_FROM = 0


python diploma/show_results.py --agents_number=$AGENTS_NUMBER --env_name=$ENV_NAME --agent=$AGENT --noise_env_step=$NOISE_ENV_STEP --metrics_number_of_elements=$METRICS_NUMBER_OF_ELEMENTS --metrics_number_of_iterations=$METRICS_NUMBER_OF_ITERATIONS --enable_exchange=$ENABLE_EXCHANGE --executions_count=$EXECUTIONS_COUNT --executions_from=$EXECUTIONS_FROM
