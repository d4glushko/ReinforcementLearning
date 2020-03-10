import argparse
import time
import numbers

from utils import str2bool, int_or_none, str_or_none
from noise_learning.utils import NoiseLearningAgents, ExchangeTypes
from noise_learning.visualizer import Visualizer


def main(arguments):
    agents_number = arguments.agents_number
    env_name = arguments.env_name
    agent = NoiseLearningAgents[arguments.agent]
    noise_env_step = arguments.noise_env_step
    noise_dropout_step = arguments.noise_dropout_step

    early_stopping = arguments.early_stopping


    exchange_type = ExchangeTypes[arguments.exchange_type]
    exchange_delta = arguments.exchange_delta
    exchange_items_reward_count = arguments.exchange_items_reward_count
    num_steps_per_episode = arguments.num_steps_per_episode


    detailed_agents_plots = arguments.detailed_agents_plots
    metrics_number_of_elements = arguments.metrics_number_of_elements
    metrics_number_of_iterations = arguments.metrics_number_of_iterations

    executions_count = arguments.executions_count
    executions_from = arguments.executions_from
    execution_date = arguments.execution_date

    visualizer = Visualizer(
        exchange_type=exchange_type, exchange_delta=exchange_delta, exchange_items_reward_count=exchange_items_reward_count, 
        agents_number=agents_number, env_name=env_name, noise_learning_agent=agent, metrics_number_of_elements=metrics_number_of_elements, 
        metrics_number_of_iterations=metrics_number_of_iterations, detailed_agents_plots=detailed_agents_plots, 
        noise_env_step=noise_env_step, noise_dropout_step=noise_dropout_step, early_stopping=early_stopping, num_steps_per_episode=num_steps_per_episode,
        executions_count=executions_count, executions_from=executions_from, execution_date=execution_date
    )

    visualizer.set_train_metrics()
    visualizer.show_train_metrics()

    if not arguments.ignore_play:
        visualizer.set_play_metrics()
        visualizer.show_play_metrics()

    input("Press <ENTER> to continue")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agents_number', type=int, default=10)
    parser.add_argument('--env_name', type=str, default='CartPole-v1')
    parser.add_argument('--agent', type=str, default='DQN')
    parser.add_argument('--noise_env_step', type=float, default=0.1)
    parser.add_argument('--noise_dropout_step', type=float, default=0.1)

    parser.add_argument('--early_stopping', type=str2bool, default=False)



    parser.add_argument('--exchange_type', type=str, default='NO')
    parser.add_argument('--exchange_delta', type=float, default=0.1)
    parser.add_argument('--exchange_items_reward_count', type=int, default=30)
    parser.add_argument('--num_steps_per_episode', type=int, default=0)

    parser.add_argument('--detailed_agents_plots', type=str2bool, default=False)
    parser.add_argument('--metrics_number_of_elements', type=int, default=100)
    parser.add_argument('--metrics_number_of_iterations', type=int, default=50)

    parser.add_argument('--ignore_play', type=str2bool, default=False)

    parser.add_argument('--executions_count', type=int_or_none)
    parser.add_argument('--executions_from', type=int_or_none)
    parser.add_argument('--execution_date', type=str_or_none)

    args = parser.parse_args()
    print(f"Called with args: {args}")
    main(args)
