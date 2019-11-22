import argparse

from noise_learning.noise_learning import NoiseLearning, NoiseLearningAgents


def main(arguments):
    agents_number = 1
    env_name = 'CartPole-v1'
    agent = NoiseLearningAgents.TEST

    noise_learning = NoiseLearning(agents_number, env_name, agent)
    noise_learning.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
