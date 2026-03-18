import gymnasium as gym
import numpy as np
import cv2

CliffEnv = gym.make("CliffWalking-v1", render_mode=None)

# Handy functions for Visuals
def initialize_frame():
    width, height = 600, 200
    img = np.ones(shape=(height, width, 3)) * 255.0
    margin_horizontal = 6
    margin_vertical = 2

    # Vertical Lines
    for i in range(13):
        img = cv2.line(img, (49 * i + margin_horizontal, margin_vertical),
                       (49 * i + margin_horizontal, 200 - margin_vertical),
                       color=(0, 0, 0), thickness=1)

    # Horizontal Lines
    for i in range(5):
        img = cv2.line(img, (margin_horizontal, 49 * i + margin_vertical),
                       (600 - margin_horizontal, 49 * i + margin_vertical),
                       color=(0, 0, 0), thickness=1)

    # Cliff Box
    img = cv2.rectangle(
        img,
        (49 * 1 + margin_horizontal + 2, 49 * 3 + margin_vertical + 2),
        (49 * 11 + margin_horizontal - 2, 49 * 4 + margin_vertical - 2),
        color=(255, 0, 255),
        thickness=-1
    )

    img = cv2.putText(
        img, "Cliff",
        (49 * 5 + margin_horizontal, 49 * 4 + margin_vertical - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
    )

    # Goal
    frame = cv2.putText(
        img, "G",
        (49 * 11 + margin_horizontal + 10, 49 * 4 + margin_vertical - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
    )

    return frame


def put_agent(img, state):
    margin_horizontal = 6
    margin_vertical = 2
    row, column = np.unravel_index(state, (4, 12))

    cv2.putText(
        img, "A",
        (49 * column + margin_horizontal + 10,
         49 * (row + 1) + margin_vertical - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
    )

    return img


state, info = CliffEnv.reset()
done = False
frame = initialize_frame()

while not done:
    frame2 = put_agent(frame.copy(), state)
    cv2.imshow("Cliff Walking", frame2)
    cv2.waitKey(250)

    action = CliffEnv.action_space.sample()

    state, reward, terminated, truncated, info = CliffEnv.step(action)

    done = terminated or truncated

CliffEnv.close()
cv2.destroyAllWindows()
