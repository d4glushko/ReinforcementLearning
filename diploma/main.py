import argparse
import time

from utils import str2bool
from noise_learning.noise_learning import NoiseLearning, NoiseLearningAgents


def main(arguments):
    agents_number = arguments.agents_number
    env_name = arguments.env_name
    agent = NoiseLearningAgents[arguments.agent]
    noise_env_step = arguments.noise_env_step
    enable_exchange = arguments.enable_exchange

    metrics_number_of_elements = arguments.metrics_number_of_elements
    metrics_number_of_iterations = arguments.metrics_number_of_iterations

    debug = arguments.debug
    use_cuda = arguments.use_cuda
    training_episodes = arguments.training_episodes

    current_execution = arguments.current_execution
    total_executions = arguments.total_executions

    show_current_result = False

    noise_learning = NoiseLearning(
        enable_exchange, training_episodes, agents_number, env_name, agent, debug, metrics_number_of_elements, 
        metrics_number_of_iterations, noise_env_step, use_cuda, 
        current_execution=current_execution, total_executions=total_executions
    )

    start = time.time()
    noise_learning.train()
    end = time.time()
    print(f"Execution time: {(end - start) / 60} minutes")

    noise_learning.save_results()

    if show_current_result:
        noise_learning.show_metrics()
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

    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument('--use_cuda', type=str2bool, default=True)
    parser.add_argument('--training_episodes', type=int, default=1000)

    parser.add_argument('--current_execution', type=int, default=1)
    parser.add_argument('--total_executions', type=int, default=1)
    args = parser.parse_args()
    print(f"Called with args: {args}")
    main(args)
