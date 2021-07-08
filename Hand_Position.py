import numpy as np
import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def get_params(hand_landmarks):
    points = {}
    parameters = ['INDEX_TIP', 'WRIST', 'MIDDLE_FINGER', 'RING_FINGER', 'PINKY_FINGER', 'THUMB']
    index_points = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    points['INDEX_TIP'] = np.array([index_points.x, index_points.y, index_points.z])
    wrist_points = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    points['WRIST'] = np.array([wrist_points.x, wrist_points.y, wrist_points.z])
    middle_finger_points = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    points['MIDDLE_FINGER'] = np.array([middle_finger_points.x, middle_finger_points.y, middle_finger_points.z])
    ring_finger_points = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    points['RING_FINGER'] = np.array([ring_finger_points.x, ring_finger_points.y, ring_finger_points.z])
    pinky_points = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    points['PINKY_FINGER'] = np.array([pinky_points.x, pinky_points.y, pinky_points.z])
    thumb_points = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    points['THUMB'] = np.array([thumb_points.x, thumb_points.y, thumb_points.z])

    hand_x_value = sum([points[i][0] for i in
                        parameters]) / len(parameters)

    hand_y_value = sum([points[i][1] for i in
                        parameters]) / len(parameters)

    return [hand_x_value, hand_y_value]


def gridify(xy_array):
    x = xy_array[0]
    y = xy_array[1]
    if x >= 0.5 and y > 0.5:
        return 'Bottom right'
    elif x < 0.5 and y >= 0.5:
        return 'Bottom left'
    elif x < 0.5 and y < 0.5:
        return 'Top left'
    else:
        return 'Top right'


def ratio_to_pixel(coordinates, image_shape):
    rows, cols, _ = image_shape
    if coordinates is None:
        return None
    return np.array(coordinates) * np.array([int(cols), int(rows)])


def label_params(frame, coordinates, text):
    if coordinates is None:
        return

    # print(centre of hand)
    cv2.putText(frame, text, (int(coordinates[0]), int(coordinates[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
