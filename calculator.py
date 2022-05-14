import cv2
import os
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import yaml
import pandas as pd

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


class GetData:

    def __init__(self, pose):
        self.pose = pose
        with open('metadata.yaml') as f:
            self.pose_metadata = yaml.safe_load(f)[pose]

    def calculate_angle(self, a, b, c):
        a = np.array(a)  # first
        b = np.array(b)  # mid
        c = np.array(c)  # last

        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
            np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)

        if angle > 180.0:
            angle = 360-angle

        return angle

    def get_pose_points(self, image):

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

            # convert to rgb image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # process image to get points
            # with mp_pose.Pose() as pose_tracker:
            result = pose.process(image=image)
            pose_landmarks = result.pose_landmarks
            pose_landmarks = [[lmk.x, lmk.y, lmk.z]
                              for lmk in pose_landmarks.landmark]

            return pose_landmarks

    def for_image(self, image):
        # try:
        points_list = self.get_pose_points(image)
        data = list()
        if 'left_thigh' in self.pose_metadata:
            data.append(self.calculate_angle(points_list[24], points_list[26], (
                points_list[26][0]-5, points_list[26][1], points_list[26][2])))

        if 'right_thigh' in self.pose_metadata:
            data.append(self.calculate_angle(points_list[23], points_list[25], (
                points_list[25][0]-5, points_list[25][1], points_list[25][2]),))

        if 'left_calf' in self.pose_metadata:
            data.append(self.calculate_angle(points_list[25], points_list[27], (
                points_list[27][0]-5, points_list[27][1], points_list[27][2])))

        if 'right_calf' in self.pose_metadata:
            data.append(self.calculate_angle(points_list[26], points_list[28], (
                points_list[28][0]-5, points_list[28][1], points_list[28][2]),))

        if 'left_side' in self.pose_metadata:
            data.append(self.calculate_angle(points_list[11], points_list[23], (
                points_list[23][0]-5, points_list[23][1], points_list[23][2])))

        if 'right_side' in self.pose_metadata:
            data.append(self.calculate_angle(points_list[12], points_list[24], (
                points_list[24][0]-5, points_list[24][1], points_list[24][2]),))

        if 'facing' in self.pose_metadata:
            if (points_list[0][0] - ((points_list[11][0] + points_list[12][0])/2)) > 0:
                data.append(1)
            else:
                data.append(0)

        if 'left_ankel' in self.pose_metadata:
            data.append(points_list[29][2])

        if 'right_ankel' in self.pose_metadata:
            data.append(points_list[30][2])

        if 'feet_knee' in self.pose_metadata:
            data.append(
                points_list[29][0]-points_list[25][0] / points_list[25][0]-points_list[30][0])

        if 'right_leg' in self.pose_metadata:
            data.append(self.calculate_angle(
                points_list[24], points_list[26], points_list[28],))

        if 'left_leg' in self.pose_metadata:
            data.append(self.calculate_angle(
                points_list[23], points_list[25], points_list[27],))

        if 'trunk_up_left' in self.pose_metadata:
            data.append(self.calculate_angle(
                points_list[13], points_list[11], points_list[23],))

        if 'trunk_up_right' in self.pose_metadata:
            data.append(self.calculate_angle(
                points_list[14], points_list[12], points_list[24],))

        if 'trunk_down_left' in self.pose_metadata:
            data.append(self.calculate_angle(
                points_list[11], points_list[23], points_list[25]))

        if 'trunk_down_right' in self.pose_metadata:
            data.append(self.calculate_angle(
                points_list[12], points_list[24], points_list[26]))

        if 'knee_hip_angle' in self.pose_metadata:
            data.append(self.calculate_angle(
                points_list[25], points_list[23], points_list[26]))

        if 'feet_knee_1' in self.pose_metadata:
            data.append(
                points_list[29][0]-points_list[25][0] / points_list[25][0]-points_list[30][0])

        if 'feet_knee_2' in self.pose_metadata:
            data.append(
                points_list[29][0]-points_list[26][0] / points_list[26][0]-points_list[30][0])

        datas = dict()
        for key,val in zip(self.pose_metadata,data):
            datas[key] = [val]
        return pd.DataFrame(datas)

        # except:
        #     return []
