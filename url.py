from django.urls import path

from . import views


app_name = 'trainer'

urlpatterns = [
    path('', views.index, name="index"),
    path('options/', views.options, name='options'),
    path('developers/', views.developers, name="developers"),
    path('contact/', views.contact, name="contact"),
    path('yoga1/<str:pose_name>/', views.yoga1, name="yoga1"),
    path('yoga1begin/<str:pose_name>', views.yoga1begin, name="yoga1begin"),
    path('video_feed_1/', views.video_feed_1, name="video_feed_1"),
    path('gen_frames/', views.gen_frames, name='gen_frames'),
]