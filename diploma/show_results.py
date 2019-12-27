import argparse
import time
import numbers

from utils import str2bool, int_or_none
from noise_learning.utils import NoiseLearningAgents
from noise_learning.visualizer import Visualizer


def main(arguments):
    agents_number = arguments.agents_number
    env_name = arguments.env_name
    agent = NoiseLearningAgents[arguments.agent]
    noise_env_step = arguments.noise_env_step
    enable_exchange = arguments.enable_exchange

    metrics_number_of_elements = arguments.metrics_number_of_elements
    metrics_number_of_iterations = arguments.metrics_number_of_iterations

    executions_count = arguments.executions_count
    executions_from = arguments.executions_from

    visualizer = Visualizer(
        enable_exchange, agents_number, env_name, agent, metrics_number_of_elements, 
        metrics_number_of_iterations, noise_env_step, executions_count, executions_from
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
    parser.add_argument('--enable_exchange', type=str2bool, default=True)

    parser.add_argument('--metrics_number_of_elements', type=int, default=100)
    parser.add_argument('--metrics_number_of_iterations', type=int, default=50)

    parser.add_argument('--executions_count', type=int_or_none)
    parser.add_argument('--executions_from', type=int_or_none)

    args = parser.parse_args()
    print(f"Called with args: {args}")
    main(args)
