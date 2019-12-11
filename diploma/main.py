import argparse
import time

from noise_learning.noise_learning import NoiseLearning, NoiseLearningAgents


def main(arguments):
    agents_number = 10
    env_name = 'CartPole-v1'
    agent = NoiseLearningAgents.DQN
    debug = False
    training_episodes = 100
    metrics_number_of_elements = 10
    metrics_number_of_iterations = 5
    noise_env_step = 0.1
    use_cuda = True

    noise_learning = NoiseLearning(
        agents_number, env_name, agent, debug, metrics_number_of_elements, metrics_number_of_iterations, noise_env_step, use_cuda
    )

    start = time.time()
    noise_learning.train(training_episodes)
    end = time.time()
    noise_learning.show_metrics()
    print(f"Execution time: {(end - start) / 60} minutes")
    input("Press <ENTER> to continue")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
