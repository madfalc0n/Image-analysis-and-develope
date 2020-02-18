# from imutils import face_utils
# import numpy as np
# import imutils
# import dlib
# import cv2
# import face_recognition #검출기 + 인식기
# import os
# from imutils import paths


# # #파라미터 초기화
# # # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# # predictor = dlib.shape_predictor("cfg_file/face_recognition/shape_predictor_68_face_landmarks.dat") #하라이크처럼 고속으로 처리되는 알고리즘 사용
# # detector = dlib.get_frontal_face_detector()



# def face(img):
#     print(img.split('/'))
#     file_name = img.split('/')[-1]
#     url = img
#     img = cv2.imread(img)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     print("그레이 변환 완료")
#     face_locations = face_recognition.face_locations(gray) # CNN 기반은 속도가 느리다,, strong classifier
#     #print("I found {} face(s) in this photograph.".format(len(face_locations)))
#     for face_location in face_locations:
#         top, right, bottom, left = face_location
#         cv2.rectangle(img, (left, top),  (right, bottom), (0, 0, 255), 3)
#     cv2.imwrite(url, img)
#     print("변환 완료")
#     return file_name