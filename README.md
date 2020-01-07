# Reinforcement Learning Master Thesis
This is a repository for a Master thesis project.

## How To Train

### Setup Virtualenv
```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run Multiple Train Executions
For the first time only:
```
chmod +x train_multiple_runs.sh
```

(Optional)
Change arguments in `train_multiple_runs.sh` file. Default arguments:
```
AGENTS_NUMBER=10
ENV_NAME='CartPole-v1'

# Available agents: DQN, A2C
AGENT='A2C'
NOISE_ENV_STEP=0.1

# Whether to give smaller eps to envs with higher noise. Applicable only for DQN currently and only for initial state (doesn't take into account exchanges)
EPS_WRT_NOISE=false

# Available types: NO, RANDOM, SMART
EXCHANGE_TYPE='SMART'
EXCHANGE_DELTA=0.1
EXCHANGE_ITEMS_REWARD_COUNT=30

WARM_UP_STEPS=30
EXCHANGE_STEPS=5

DEBUG=false
USE_CUDA=false
TRAINING_EPISODES=5000
PLAY_EPISODES=500
IGNORE_PLAY=false
```

Run multiple train executions (pass parameter for train executions count. Default 10 will be used if no parameter passed):
```
./train_multiple_runs.sh 5
```

Or run in a background process:
```
nohup ./train_multiple_runs.sh > log.out &
```

## Show Results
For the first time only:
```
chmod +x show_aggregated_results.sh
```

(Optional)
Change arguments in `show_aggregated_results.sh` file to determine the executions set that will be used to show the results. Default arguments:
```
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

IGNORE_PLAY=false

# EXECUTIONS_COUNT=1
# EXECUTIONS_FROM=0
# EXECUTION_DATE='2020-01-03_10:22:52'
```

Show results:
```
./show_aggregated_results.sh
```