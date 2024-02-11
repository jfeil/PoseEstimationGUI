#!/usr/bin/env python3
import sys
from collections import defaultdict

import numpy as np
import pandas
import cv2
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QScrollArea
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib import pyplot as plt

skeleton = ([0, 2], [9, 8], [8, 6], [9, 1], [4, 5], [9, 0], [2, 3], [1, 4], [8, 7])


class PoseCanvas(FigureCanvas):
    def __init__(self, data, player_key, parent=None):
        self.fig, self.ax = plt.subplots()
        super().__init__(self.fig)

        self.setParent(parent)
        self.setParent(parent)

        self.data = data
        self.fig.suptitle(f'Player {player_key}', fontsize=16)
        self.min = min(self.data.FrameNo)
        self.max = max(self.data.FrameNo)

    def valid(self, frame_index):
        return self.min <= frame_index <= self.max


class RotationCanvas(PoseCanvas):
    def __init__(self, data, player_key, current_frame, parent=None):
        super().__init__(data, player_key, parent)
        self.red_bar = self.ax.axvline(x=current_frame, color='red', linewidth=2)

    def plot(self):
        self.ax.plot(self.data.FrameNo.to_numpy(), self.data.orientation.to_numpy())
        self.draw()

    def mark_timestamp(self, frame_index):
        self.red_bar.set_data([frame_index, frame_index], [0, 1])
        self.draw()


class PlayerDisplay(PoseCanvas):
    def __init__(self, data, player_key, parent=None):
        super().__init__(data, player_key, parent)

    def plot(self, frame_index):
        self.ax.clear()
        pred = self.data[self.data.FrameNo == frame_index].to_numpy()[0]
        centers = np.reshape(pred[4:], (-1, 3))
        # self.ax[0].scatter(centers[:, 0], centers[:, 1], c='r')

        for x, y, score in centers[:]:
            self.ax.annotate(f"{score*100:.1f}", (x, y))
        for p1, p2 in skeleton:
            coords = np.vstack((centers[p1, :2], centers[p2, :2]))
            self.ax.plot(coords[:, 0], coords[:, 1], 'bo--')
        self.draw()


class AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multiple Plots in PyQt")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.scroll_area = QScrollArea(self.central_widget)
        self.scroll_area.setWidgetResizable(True)
        self.layout.addWidget(self.scroll_area)

        self.scroll_content = QWidget(self.scroll_area)
        self.scroll_area.setWidget(self.scroll_content)
        self.scroll_layout = QVBoxLayout(self.scroll_content)

        self.plot_widgets = {}

    def add_plot(self, player_key: int, plot_widget: PoseCanvas):
        self.scroll_layout.addWidget(plot_widget)
        self.plot_widgets[player_key] = plot_widget

    def delete_plot(self, player_key: int):
        self.scroll_layout.removeWidget(self.plot_widgets[player_key])
        plt.close(self.plot_widgets.pop(player_key).fig)


class VideoPlayer:
    def __init__(self, video_path, annotations):
        self.video_path = video_path
        self.annotations = pandas.read_csv(annotations)

        self.gui = AppWindow()
        self.gui.show()

        self.players = {}

        self.cap = cv2.VideoCapture(self.video_path)
        self.threshold = 0.5

        self.video_length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cv2.namedWindow('Frame', cv2.WINDOW_GUI_EXPANDED)
        cv2.createTrackbar('frame_nr', 'Frame', 0, self.video_length, self.set_displayed_frame)
        cv2.createTrackbar('threshold', 'Frame', int(self.threshold * 100), 100, self.set_threshold)
        video_is_running = True
        self.single_update = False

        self.draw = True
        self.draw_gui = True

        while self.cap.isOpened():
            # Capture frame-by-frame
            frame_nr = self.cap.get(cv2.CAP_PROP_POS_FRAMES) + 1
            if video_is_running or self.single_update:
                self.single_update = False
                ret, frame = self.cap.read()
                if ret:
                    if self.draw:
                        frame = self.draw_overlays(frame, frame_nr)
                    # Display the resulting frame
                    cv2.imshow('Frame', frame)
                    # Press Q on keyboard to  exit
            key = cv2.waitKey(25)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('d'):
                self.draw = not self.draw
            elif key & 0xFF == ord('g'):
                self.draw_gui = not self.draw_gui
            elif key & 0xFF == ord('p'):
                if video_is_running:
                    cv2.setTrackbarPos('frame_nr', 'Frame', int(frame_nr))
                video_is_running = not video_is_running

            # Break the loop
        cv2.destroyAllWindows()

    def set_draw(self, value, source=None):
        self.draw = value

    def set_threshold(self, threshold):
        self.threshold = threshold / 100

    def set_displayed_frame(self, frame_nr):
        self.single_update = True
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nr - 1)

    def draw_overlays(self, frame, frame_nr):
        if self.draw_gui:
            delete_list = []
            for key, (rotation_dia, annot_dia) in self.players.items():
                if not rotation_dia.valid(frame_nr):
                    delete_list += [key]
                else:
                    rotation_dia.mark_timestamp(int(frame_nr))

            for key in delete_list:
                self.gui.delete_plot(key)
                self.players.pop(key)

            if len(delete_list) > 0:
                self.gui.scroll_area.repaint()

        frame = cv2.putText(frame, f"{int(frame_nr):05d}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        preds = self.annotations[self.annotations.FrameNo == frame_nr]
        for index, row in preds.iterrows():
            player_key = int(row.PlayerKey)
            pos = row.to_numpy()[4:].reshape([-1, 3])
            frame = cv2.putText(frame, f"{int(row.PlayerKey)}",
                                (int(pos.max(axis=0)[0]) + 10, int(pos.min(axis=0)[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            frame = draw_skeleton(frame, row.to_numpy(), threshold=self.threshold)
            frame = draw_arrow(frame, row.orientation, np.array([row['Hip x'], row['Hip y']]))
            if self.draw_gui:
                if row.PlayerKey not in self.players:
                    player_data = self.annotations[self.annotations.PlayerKey == row.PlayerKey]
                    if len(player_data) < 10:
                        continue
                    rot_canvas = RotationCanvas(player_data, player_key, int(frame_nr))
                    rot_canvas.plot()
                    player_canvas = PlayerDisplay(player_data, player_key)

                    self.players[player_key] = (rot_canvas, player_canvas)
                    self.gui.add_plot(player_key, rot_canvas)
                self.players[player_key][0].mark_timestamp(frame_nr)
                self.players[player_key][1].plot(frame_nr)
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
    app = QApplication(sys.argv)
    player = VideoPlayer('/home/jfeil/IDP/raw_data/Tennis/DJI_0170.MP4',
                         '/home/jfeil/DroneTracking/Results/run_4/pose_tracks.csv')
