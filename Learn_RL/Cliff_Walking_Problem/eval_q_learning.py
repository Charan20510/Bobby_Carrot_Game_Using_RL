import gymnasium
import numpy as np
import pickle as pkl
import cv2

from show_pannel import initialize_frame, put_agent


def epsi_greedy_policy(state, epsi=0.0):
    action = int(np.argmax(q_table[state]))
    if np.random.random() <= epsi:
        action = int(np.random.randint(low=0, high=1))
    return action


cliffEnv = gymnasium.make("CliffWalking-v0")


q_table = pkl.load(open("q_learning_q_table.pkl", "rb"))

# Parameters
NUM_EPISODES = 5

for episode in range(NUM_EPISODES):
    done = False
    total_reward = 0
    episode_length = 0
    state, info = cliffEnv.reset()
    frame = initialize_frame()
    while not done:
        frame2 = put_agent(frame.copy(), state=state)
        cv2.imshow("Cliff Walking", frame2)
        cv2.waitKey(250)

        action = epsi_greedy_policy(state=state)
        state, reward, terminated, truncated, info = cliffEnv.step(action=action)
        total_reward += reward
        episode_length += 1
        done = terminated or truncated
    print("Episode = ", episode, "| Episode Length = ", episode_length, "| Total Reward = ", total_reward)

cliffEnv.close()
cv2.destroyAllWindows()
