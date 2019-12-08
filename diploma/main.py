import argparse

from noise_learning.noise_learning import NoiseLearning, NoiseLearningAgents


def main(arguments):
    agents_number = 1
    env_name = 'CartPole-v1'
    agent = NoiseLearningAgents.TEST
    debug = False
    training_episodes = 1000
    metrics_number_of_elements = 100
    metrics_number_of_iterations = 25

    noise_learning = NoiseLearning(agents_number, env_name, agent, debug, metrics_number_of_elements, metrics_number_of_iterations)
    noise_learning.train(training_episodes)
    noise_learning.show_metrics()
    input("Press <ENTER> to continue")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
