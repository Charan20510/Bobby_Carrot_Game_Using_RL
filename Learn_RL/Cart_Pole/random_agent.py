import cv2
import gymnasium
import numpy as np

cartEnv = gymnasium.make("CartPole-v1", render_mode="rgb_array")

for episode in range(5):
    done = False
    status, info = cartEnv.reset()
    while not done:
        frame = cartEnv.render()
        cv2.imshow("Cart Pole", frame)
        cv2.waitKey(100)
        action = np.random.randint(0, cartEnv.action_space.n)
        state, reward, terminated, truncated, info = cartEnv.step(action=action)
        done = terminated or truncated
cartEnv.close()
cv2.destroyAllWindows()
