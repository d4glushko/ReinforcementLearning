import argparse
import time

from utils import str2bool
from noise_learning.utils import NoiseLearningAgents, ExchangeTypes
from noise_learning.noise_learning import NoiseLearning


def main(arguments):
    agents_number = arguments.agents_number
    env_name = arguments.env_name
    agent = NoiseLearningAgents[arguments.agent]
    noise_env_step = arguments.noise_env_step

    exchange_type = ExchangeTypes[arguments.exchange_type]
    exchange_delta = arguments.exchange_delta
    exchange_items_reward_count = arguments.exchange_items_reward_count

    warm_up_steps = arguments.warm_up_steps
    exchange_steps = arguments.exchange_steps

    debug = arguments.debug
    use_cuda = arguments.use_cuda
    training_episodes = arguments.training_episodes

    date = arguments.date
    current_execution = arguments.current_execution
    total_executions = arguments.total_executions

    noise_learning = NoiseLearning(
        exchange_type=exchange_type, exchange_delta=exchange_delta, exchange_items_reward_count=exchange_items_reward_count, 
        training_episodes=training_episodes, agents_number=agents_number, env_name=env_name, noise_learning_agent=agent, debug=debug, 
        noise_env_step=noise_env_step, use_cuda=use_cuda, warm_up_steps=warm_up_steps, exchange_steps=exchange_steps, 
        date=date, current_execution=current_execution, total_executions=total_executions
    )

    start = time.time()
    noise_learning.train()
    end = time.time()
    print(f"Execution time: {(end - start) / 60} minutes")

    noise_learning.save_results()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agents_number', type=int, default=10)
    parser.add_argument('--env_name', type=str, default='CartPole-v1')
    parser.add_argument('--agent', type=str, default='DQN')
    parser.add_argument('--noise_env_step', type=float, default=0.1)

    parser.add_argument('--exchange_type', type=str, default='NO')
    parser.add_argument('--exchange_delta', type=float, default=0.1)
    parser.add_argument('--exchange_items_reward_count', type=int, default=30)

    parser.add_argument('--warm_up_steps', type=int, default=30)
    parser.add_argument('--exchange_steps', type=int, default=5)

    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument('--use_cuda', type=str2bool, default=True)
    parser.add_argument('--training_episodes', type=int, default=1000)

    parser.add_argument('--date', type=int, required=True)
    parser.add_argument('--current_execution', type=int, default=1)
    parser.add_argument('--total_executions', type=int, default=1)
    args = parser.parse_args()
    print(f"Called with args: {args}")
    main(args)
