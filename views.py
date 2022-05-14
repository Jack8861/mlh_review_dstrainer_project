from curses.ascii import HT
from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from joblib import load
from django.http import StreamingHttpResponse
from .models import Pose
import cv2
import cv2 as cv
import os
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import yaml
import pandas as pd
from dstrainer.settings import BASE_DIR

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
# Create your views here.


def gen_frames(yoga_name="planks"):
    print("planks")
    print(os.path.dirname(__file__))
    data_fetcher = GetData(yoga_name)
    module_dir = os.path.dirname(__file__)  
    file_path = os.path.join(module_dir, 'planks.joblib') 
    model = load(file_path)
    try:
        data_fetcher = GetData('planks')

        model = load('{}.joblib'.format(yoga_name))
    except:
        print("some error")
    cap = cv.VideoCapture(0)

    while (cap.isOpened()):

        ret, frame = cap.read()
        if ret == True:
            try:
                frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
                img = cv.imencode('.jpg', frame)[1].tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')
                data = data_fetcher.for_image(image=frame)
                print(model.predict(data))
            except Exception as e:
                print("error while prediction", e)
                continue
        else:
            break

    img = cv.imread(r"/home/rahim/Desktop/Final_Year_project/8sem/download.jpeg")
    img = cv.imencode('.jpg', img)[1].tobytes()
    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')



# welcome page
def index(request):
    template = loader.get_template('trainer/index.html')
    return HttpResponse(template.render())

# options page
# @ app.route('/options')
def options(request):

    pose_data = Pose.objects.all().values_list('pose_name', 'details', 'image')
    GetData('planks')


    pose_list = []
    for i in range(3,len(pose_data)+1,3):
        pose_list.append(pose_data[i-3:i])
    extra = len(pose_data)%3 
    if extra != 0:
        pose_list.append(pose_data[int(len(pose_data)/3)*3:])
    
    template = loader.get_template('trainer/options.html')
    context = {
        'pose_list':pose_list,
    }
    return HttpResponse(template.render(context, request))






# developers page
# @ app.route('/developers')
def developers(request):
    template = loader.get_template('trainer/developers.html')
    return HttpResponse(template.render())

# contact page
# @ app.route('/contact')
def contact(request):
    template = loader.get_template('trainer/contact.html')
    return HttpResponse(template.render())

# yoga1 page
# @ app.route('/yoga1')
def yoga1(request, pose_name):
    obj = Pose.objects.get(pose_name=pose_name)
    template = loader.get_template('trainer/yoga1.html')
    context = {
        "pose": obj,
    }
    return HttpResponse(template.render(context))

# yoga1begin page
# @ app.route('/yoga1begin')
def yoga1begin(request, pose_name):
    obj = Pose.objects.get(pose_name=pose_name)
    template = loader.get_template('trainer/yoga1begin.html')
    context = {
        "pose": obj,
    }
    return HttpResponse(template.render(context))

# videofeed page
# @ app.route('/video_feed_1')
def video_feed_1(request):
    print("video_feed_1")
    template = loader.get_template('trainer/yoga1.html')
    return StreamingHttpResponse(gen_frames("planks"), content_type='multipart/x-mixed-replace; boundary=frame')

# def video_feed_1():
#     return Response(gen_frames("yoga1"), mimetype='multipart/x-mixed-replace; boundary=frame')


# def index(request):
#     latest_question_list = Question.objects.all()
#     output = ', '.join([q.question_text for q in latest_question_list])

#     template = loader.get_template('polls/index.html')
#     context = {
#         'latest_question_list' : latest_question_list
#     }
#     return HttpResponse(template.render(context, request))

# def detail(request, question_id):
#     try:
#         question = Question.objects.get(pk=1)
#     except Question.DoesNotExist:
#         raise Http404("Question does not exist")

#     template = loader.get_template('polls/detail.html')
#     context = {
#         'question' : question
#     }
#     return HttpResponse(template.render(context, request))


# def results(request, question_id):
#     response = "you are looking at results for {}".format(question_id)
#     return HttpResponse(response)


# def vote(request, question_id):
#     return HttpResponse("You are looking at votes {}".format(question_id))
























class GetData:

    def __init__(self, pose_name):
        self.pose_name = pose_name
        print(self.pose_name)
        self.pose = Pose.objects.get(pose_name=self.pose_name)
        self.pose_metadata = list()
        # file_path = self.your_view_function()
        print(self.pose.classifier.url)
        # model = load('..%s'%(self.pose.classifier.url))
        # file_path = os.path.join(BASE_DIR, self.pose.classifier.url[1:])
        # model = load(file_path)
        # print("path is", file_path)
        # print("path is", file_path)

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
        # TODO: train the models without column names and remove pose_metadata below

        points_list = self.get_pose_points(image)
        data = list()

        if self.pose.left_thigh == True:
            self.pose_metadata.append('left_thigh')
            data.append(self.calculate_angle(points_list[24], points_list[26], (
                points_list[26][0]-5, points_list[26][1], points_list[26][2])))

        if  self.pose.right_thigh == True:
            self.pose_metadata.append('right_thigh')
            data.append(self.calculate_angle(points_list[23], points_list[25], (
                points_list[25][0]-5, points_list[25][1], points_list[25][2]),))

        if  self.pose.left_calf == True:
            self.pose_metadata.append('left_calf')
            data.append(self.calculate_angle(points_list[25], points_list[27], (
                points_list[27][0]-5, points_list[27][1], points_list[27][2])))

        if  self.pose.right_calf == True:
            self.pose_metadata.append('right_calf')
            data.append(self.calculate_angle(points_list[26], points_list[28], (
                points_list[28][0]-5, points_list[28][1], points_list[28][2]),))

        if self.pose.left_side == True:
            self.pose_metadata.append('left_side')
            data.append(self.calculate_angle(points_list[11], points_list[23], (
                points_list[23][0]-5, points_list[23][1], points_list[23][2])))

        if  self.pose.right_side == True:
            self.pose_metadata.append('right_side')
            data.append(self.calculate_angle(points_list[12], points_list[24], (
                points_list[24][0]-5, points_list[24][1], points_list[24][2]),))

        if  self.pose.facing == True:
            self.pose_metadata.append('facing')
            if (points_list[0][0] - ((points_list[11][0] + points_list[12][0])/2)) > 0:
                data.append(1)
            else:
                data.append(0)

        if self.pose.left_ankel == True:
            self.pose_metadata.append('left_ankel')
            data.append(points_list[29][2])

        if self.pose.right_ankel == True:
            self.pose_metadata.append('right_ankel')
            data.append(points_list[30][2])

        if self.pose.feet_knee == True:
            self.pose_metadata.append('feet_knee')
            data.append(
                points_list[29][0]-points_list[25][0] / points_list[25][0]-points_list[30][0])

        if  self.pose.right_leg == True:
            self.pose_metadata.append('right_leg')
            data.append(self.calculate_angle(
                points_list[24], points_list[26], points_list[28],))

        if self.pose.left_leg == True:
            self.pose_metadata.append('left_leg')
            data.append(self.calculate_angle(
                points_list[23], points_list[25], points_list[27],))

        if self.pose.trunk_up_left == True:
            self.pose_metadata.append('trunk_up_left')
            data.append(self.calculate_angle(
                points_list[13], points_list[11], points_list[23],))

        if self.pose.trunk_up_right == True:
            self.pose_metadata.append('trunk_up_right')
            data.append(self.calculate_angle(
                points_list[14], points_list[12], points_list[24],))

        if  self.pose.trunk_down_left == True:
            self.pose_metadata.append('trunk_down_left')
            data.append(self.calculate_angle(
                points_list[11], points_list[23], points_list[25]))

        if self.pose.trunk_down_right == True:
            self.pose_metadata.append('trunk_down_right')
            data.append(self.calculate_angle(
                points_list[12], points_list[24], points_list[26]))

        if  self.pose.knee_hip_angle == True:

            print("\n\n\n\n\n\n Kneee\n\n\n\n\n\n")
            self.pose_metadata.append('knee_hip_angle')
            data.append(self.calculate_angle(
                points_list[25], points_list[23], points_list[26]))

        if  self.pose.feet_knee_1 == True:
            self.pose_metadata.append('feet_knee_1')
            data.append(
                points_list[29][0]-points_list[25][0] / points_list[25][0]-points_list[30][0])

        if  self.pose.feet_knee_2 == True:
            self.pose_metadata.append('feet_knee_2')
            data.append(
                points_list[29][0]-points_list[26][0] / points_list[26][0]-points_list[30][0])

        datas = dict()
        # df = pd.DataFrame(data).values.reshape(1,len(data))
        # df = pd.DataFrame(df)
        # print('columns')
        # for i in df.columns:
        #     print(i)
        for key,val in zip(self.pose_metadata,data):
            datas[key] = [val]
        return pd.DataFrame(datas)

