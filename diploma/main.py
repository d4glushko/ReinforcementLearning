import argparse

from noise_learning import NoiseLearning


def main(arguments):
    agents_number = 5
    env_name = 'CartPole-v1'
    noise_learning = NoiseLearning(agents_number, env_name)
    noise_learning.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
