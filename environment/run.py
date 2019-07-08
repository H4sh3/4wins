import gym
import gym_colonizer


def random_agent(episodes=10000):
	env = gym.make("Colonizer-v0")
	env.reset()
	env.render()
	for e in range(episodes):
		env.render()

if __name__ == "__main__":
    random_agent()