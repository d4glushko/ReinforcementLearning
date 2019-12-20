import argparse
import time

from noise_learning.noise_learning import NoiseLearning, NoiseLearningAgents


def main(arguments):
    agents_number = 2
    env_name = 'CartPole-v1'
    agent = NoiseLearningAgents.DQN
    noise_env_step = 0.1

    metrics_number_of_elements = 4
    metrics_number_of_iterations = 2

    # TODO: Refactor to not set these unneeded settings for showing results
    debug = False
    use_cuda = True
    training_episodes = 10

    noise_learning = NoiseLearning(
        training_episodes, agents_number, env_name, agent, debug, metrics_number_of_elements, 
        metrics_number_of_iterations, noise_env_step, use_cuda, ignore_training_setup=True
    )

    noise_learning.set_metrics()
    noise_learning.show_metrics()

    input("Press <ENTER> to continue")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
