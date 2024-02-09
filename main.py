#!/usr/bin/env python3
from collections import defaultdict

import numpy as np
import pandas
import cv2
from matplotlib import pyplot as plt

skeleton = ([0, 2], [9, 8], [8, 6], [9, 1], [4, 5], [9, 0], [2, 3], [1, 4], [8, 7])


class PlayerDisplay:
    def __init__(self, data, player_key, current_frame):
        self.fig, self.ax = plt.subplots(2, 1)
        self.data = data
        self.fig.suptitle(f'Player {player_key}', fontsize=16)
        self.min = min(self.data.FrameNo)
        self.max = max(self.data.FrameNo)
        self.draw_rotation()
        self.fig.show()
        self.red_bar = self.ax[1].axvline(x=current_frame, color='red', linewidth=2)

        self.draw_call = False

    def valid(self, frame_index):
        return self.min <= frame_index <= self.max

    def mark_timestamp(self, frame_index):
        self.red_bar.set_data([frame_index, frame_index], [0, 1])
        self.draw_call = True

    def draw_rotation(self):
        self.ax[1].plot(self.data.FrameNo.to_numpy(), self.data.orientation.to_numpy())

    def draw_person(self, frame_index):
        self.ax[0].clear()
        pred = self.data[self.data.FrameNo == frame_index].to_numpy()[0]
        centers = np.reshape(pred[4:], (-1, 3))
        # self.ax[0].scatter(centers[:, 0], centers[:, 1], c='r')

        for x, y, score in centers[:]:
            self.ax[0].annotate(f"{score*100:.1f}", (x, y))
        for p1, p2 in skeleton:
            coords = np.vstack((centers[p1, :2], centers[p2, :2]))
            self.ax[0].plot(coords[:, 0], coords[:, 1], 'bo--')
        self.draw_call = True

    def draw(self):
        if self.draw_call:
            self.fig.canvas.draw()
            self.draw_call = False

    def __del__(self):
        plt.close(self.fig)


class VideoPlayer:
    def __init__(self, video_path, annotations):
        self.video_path = video_path
        self.annotations = pandas.read_csv(annotations)

        self.players = {}

        self.cap = cv2.VideoCapture(self.video_path)
        self.threshold = 0.5

        self.video_length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cv2.namedWindow('Frame', cv2.WINDOW_GUI_EXPANDED)
        cv2.createTrackbar('frame_nr', 'Frame', 0, self.video_length, self.set_displayed_frame)
        cv2.createTrackbar('threshold', 'Frame', int(self.threshold * 100), 100, self.set_threshold)
        video_is_running = True
        self.single_update = False

        while self.cap.isOpened():
            # Capture frame-by-frame
            frame_nr = self.cap.get(cv2.CAP_PROP_POS_FRAMES) + 1
            if video_is_running or self.single_update:
                self.single_update = False
                ret, frame = self.cap.read()
                if ret:
                    frame = self.draw_overlays(frame, frame_nr)
                    # Display the resulting frame
                    cv2.imshow('Frame', frame)
                    # Press Q on keyboard to  exit
            key = cv2.waitKey(25)
            if key & 0xFF == ord('q'):
                break
            if key & 0xFF == ord('p'):
                if video_is_running:
                    cv2.setTrackbarPos('frame_nr', 'Frame', int(frame_nr))
                video_is_running = not video_is_running

            # Break the loop
        cv2.destroyAllWindows()

    def set_threshold(self, threshold):
        self.threshold = threshold / 100

    def set_displayed_frame(self, frame_nr):
        self.single_update = True
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nr - 1)

    def draw_overlays(self, frame, frame_nr):
        delete_list = []
        for key, value in self.players.items():
            if not value.valid(frame_nr):
                delete_list += [key]
            else:
                value.mark_timestamp(int(frame_nr))
            value.draw()

        for key in delete_list:
            self.players.pop(key)

        frame = cv2.putText(frame, f"{int(frame_nr):05d}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        preds = self.annotations[self.annotations.FrameNo == frame_nr]
        for index, row in preds.iterrows():
            pos = row.to_numpy()[4:].reshape([-1, 3])
            frame = cv2.putText(frame, f"{int(row.PlayerKey)}",
                                (int(pos.max(axis=0)[0]) + 10, int(pos.min(axis=0)[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            frame = draw_skeleton(frame, row.to_numpy(), threshold=self.threshold)
            frame = draw_arrow(frame, row.orientation, np.array([row['Hip x'], row['Hip y']]))

            if row.PlayerKey not in self.players:
                self.players[int(row.PlayerKey)] = PlayerDisplay(
                    self.annotations[self.annotations.PlayerKey == row.PlayerKey], int(row.PlayerKey), int(frame_nr))
            self.players[int(row.PlayerKey)].draw_person(frame_nr)
        return frame


def array_to_point(array):
    return int(array[0]), int(array[1])


left_color = (255, 0, 219)
center_color = (0, 228, 255)
right_color = (0, 0, 255)

colors = [
    left_color,
    right_color,
    left_color,
    left_color,
    right_color,
    right_color,
    left_color,
    right_color,
    center_color,
    center_color
]


def draw_skeleton(frame, pred, threshold):
    centers = [array_to_point(point) for point in np.reshape(pred[4:], (-1, 3))]
    scores = [point[2] for point in np.reshape(pred[4:], (-1, 3))]
    for i, point in enumerate(centers):
        if scores[i] > threshold:
            frame = cv2.circle(frame, point, 2, colors[i])
    for p1, p2 in skeleton:
        if scores[p1] > threshold and scores[p2] > threshold:
            frame = cv2.line(frame, centers[p1], centers[p2], 0)
    return frame


def draw_arrow(frame, rotation, origin, length=50, thickness=4, color=(255, 0, 0)):
    rotation_matrix = np.array([[np.cos(rotation), -np.sin(rotation)],
                                [np.sin(rotation), np.cos(rotation)]])
    # Apply rotation to the vector
    vector = np.dot(rotation_matrix, np.array((1, 0))) * length
    if np.linalg.norm(vector) == 0:
        return
    frame = cv2.arrowedLine(frame,
                            array_to_point(origin),
                            array_to_point(origin + (vector[1], -vector[0])),
                            color,
                            thickness)
    return frame


if __name__ == '__main__':
    player = VideoPlayer('/home/jfeil/IDP/raw_data/Tennis/DJI_0170.MP4',
                         '/home/jfeil/DroneTracking/Results/run_4/pose_tracks.csv')
