import gym
import gym_colonizer


def random_agent(episodes=100):
	env = gym.make("Colonizer-v0")
	env.reset()
	env.render()
	for e in range(episodes):
		action = env.action_space.sample()
		state, reward, done, _ = env.step(action)
		env.render()
		print(reward)
		if done:
			break

if __name__ == "__main__":
    random_agent()