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
AGENT='DQN'
NOISE_ENV_STEP=0.1
ENABLE_EXCHANGE=true

DEBUG=false
USE_CUDA=true
TRAINING_EPISODES=1000
```

Run multiple train executions (pass parameter for train executions count. Default 10 will be used if no parameter passed):
```
./train_multiple_runs.sh 5
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
AGENT='DQN'
NOISE_ENV_STEP=0.1
ENABLE_EXCHANGE=true

METRICS_NUMBER_OF_ELEMENTS=100
METRICS_NUMBER_OF_ITERATIONS=50

# EXECUTIONS_COUNT=1
# EXECUTIONS_FROM=0
```

Show results:
```
./show_aggregated_results.sh
```