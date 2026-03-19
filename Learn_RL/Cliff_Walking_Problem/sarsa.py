import gymnasium
import numpy as np
import pickle as pkl

cliffEnv = gymnasium.make("CliffWalking-v0")

q_table = np.zeros(shape=(48, 4))

def epsi_greedy_policy(state, epsi=0.0):
    action = int(np.argmax(q_table[state]))
    if np.random.random() <= epsi:
        action = int(np.random.randint(low=0, high=1))
    return action


# Parameters
EPSILON=0.1
ALPHA=0.1
GAMMA=0.9
NUM_EPISODES=500


for episode in range(NUM_EPISODES):
    done = False
    total_reward = 0
    episode_length = 0
    state, info = cliffEnv.reset()
    action = epsi_greedy_policy(state=state, epsi=EPSILON)

    while not done:
        next_state, reward, terminated, truncated, info = cliffEnv.step(action=action)
        next_action = epsi_greedy_policy(state=next_state, epsi=EPSILON)

        q_table[state][action] += ALPHA * (reward + GAMMA * q_table[next_state][next_action] - q_table[state][action])

        state = next_state
        action = next_action
        total_reward += reward
        episode_length += 1
        done = terminated or truncated
    print("Episode = ", episode, "| Episode Length = ", episode_length, "| Total Reward = ", total_reward)

cliffEnv.close()

pkl.dump(q_table, open("sarsa_q_table.pkl", "wb"))
print("Training Complete. Q Table Saved!")
