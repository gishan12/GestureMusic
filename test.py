import cv2
import mediapipe as mp
import numpy as np
import Hand_Position as hp
import music
import cProfile
import re

def main():

    counter = 0
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # For static images:

    comp = music.composition(tempo = 180, volume = 1)

    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = hands.process(image)

            if counter % 8 == 0:
            # Draw the hand annotations on the image.
                image.flags.writeable = True
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:

                        coordinates = hp.get_params(hand_landmarks)

                        # FOR ADVAIT !!! the points [x,y] come scaled [0,1] here so how could we run a small sound gennerating function on this?

                        grid_values = hp.gridify(coordinates)

                        note = hp.location_to_note(grid_values)

                        comp.play_note(note)
                        #comp.export_full()

                        # grid values just gives the discrete string based on coordiates ^

                        coordinates_absolute = hp.ratio_to_pixel(coordinates, image.shape)

                        # FOR ADVAIT !!! the points [x,y] come in terms of pixels here so which [x,y] pair would be better?

                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    hp.label_params(image, coordinates_absolute, grid_values)
                # the functions above are responsible for adding the markers and the text to the image in that order.
            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()


#if __name__ == '__main__':
main()

cProfile.run('main()')