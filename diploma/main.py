import argparse
import time

from noise_learning.noise_learning import NoiseLearning, NoiseLearningAgents


def main(arguments):
    agents_number = 10
    env_name = 'CartPole-v1'
    agent = NoiseLearningAgents.DQN
    noise_env_step = 0.1

    metrics_number_of_elements = 10
    metrics_number_of_iterations = 5

    debug = False
    use_cuda = True
    training_episodes = 100

    noise_learning = NoiseLearning(
        training_episodes, agents_number, env_name, agent, debug, metrics_number_of_elements, 
        metrics_number_of_iterations, noise_env_step, use_cuda
    )

    start = time.time()
    noise_learning.train()
    end = time.time()
    noise_learning.show_metrics()
    noise_learning.save_results()
    print(f"Execution time: {(end - start) / 60} minutes")
    input("Press <ENTER> to continue")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
