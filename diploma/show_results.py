import argparse
import time
import numbers

from utils import str2bool, int_or_none
from noise_learning.utils import NoiseLearningAgents, ExchangeTypes
from noise_learning.visualizer import Visualizer


def main(arguments):
    agents_number = arguments.agents_number
    env_name = arguments.env_name
    agent = NoiseLearningAgents[arguments.agent]
    noise_env_step = arguments.noise_env_step

    exchange_type = ExchangeTypes[arguments.exchange_type]
    exchange_delta = arguments.exchange_delta
    exchange_items_reward_count = arguments.exchange_items_reward_count

    detailed_agents_plots = arguments.detailed_agents_plots
    metrics_number_of_elements = arguments.metrics_number_of_elements
    metrics_number_of_iterations = arguments.metrics_number_of_iterations

    executions_count = arguments.executions_count
    executions_from = arguments.executions_from

    visualizer = Visualizer(
        exchange_type=exchange_type, exchange_delta=exchange_delta, exchange_items_reward_count=exchange_items_reward_count, 
        agents_number=agents_number, env_name=env_name, noise_learning_agent=agent, metrics_number_of_elements=metrics_number_of_elements, 
        metrics_number_of_iterations=metrics_number_of_iterations, detailed_agents_plots=detailed_agents_plots, 
        noise_env_step=noise_env_step, executions_count=executions_count, executions_from=executions_from
    )

    visualizer.set_metrics()
    visualizer.show_metrics()

    input("Press <ENTER> to continue")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agents_number', type=int, default=10)
    parser.add_argument('--env_name', type=str, default='CartPole-v1')
    parser.add_argument('--agent', type=str, default='DQN')
    parser.add_argument('--noise_env_step', type=float, default=0.1)

    parser.add_argument('--exchange_type', type=str, default='NO')
    parser.add_argument('--exchange_delta', type=float, default=0.1)
    parser.add_argument('--exchange_items_reward_count', type=int, default=30)

    parser.add_argument('--detailed_agents_plots', type=str2bool, default=False)
    parser.add_argument('--metrics_number_of_elements', type=int, default=100)
    parser.add_argument('--metrics_number_of_iterations', type=int, default=50)

    parser.add_argument('--executions_count', type=int_or_none)
    parser.add_argument('--executions_from', type=int_or_none)

    args = parser.parse_args()
    print(f"Called with args: {args}")
    main(args)
