import gymnasium as gym
import numpy as np
import cv2

from show_pannel import initialize_frame, put_agent

cliffEnv = gym.make('CliffWalking-v0')

done = False
state, info = cliffEnv.reset()
frame = initialize_frame()
while not done:
    frame2 = put_agent(frame.copy(), state)
    cv2.imshow("Cliff Walking", frame2)
    cv2.waitKey(250)
    action = int(np.random.randint(low=0, high=4))
    print(state, "==>", ["Up", "Down", "Left", "Right"][action])
    state, reward, terminated, truncated, info = cliffEnv.step(action)
    done = terminated or truncated
cliffEnv.close()
cv2.destroyAllWindows()
