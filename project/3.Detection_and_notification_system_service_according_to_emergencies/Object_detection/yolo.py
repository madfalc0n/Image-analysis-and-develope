import cv2
import argparse
import numpy as np
import os.path
import os
import time

# 파라미터 초기화
# 객체인지 아닌지 구분하기 위한 변수
# Confidence threshold
confThreshold = 0.5 
# Non-maximum suppression threshold 
nmsThreshold = 0.3  

# 해상도를 지정하기 위한 변수, 모델에 입력되는 사이즈
# Width of network's input image
inpWidth = 416  
# Height of network's input image
inpHeight = 416  


# # Load names of classes
# #print("my path is ", os.getcwd())
# #classesFile = "Object_detection/cfg_file/one.names"
# classesFile = "cfg_file/coco.names"
# classes = None
# with open(classesFile, 'rt') as f:
#     classes = f.read().rstrip('\n').split('\n')
# #print(classes)
# # Give the configuration and weight files for the model and load the network using them.
# #modelConfiguration = "Object_detection/cfg_file/one.cfg"
# modelConfiguration = "cfg_file/yolov3-tiny.cfg"
# #modelWeights = "Object_detection/cfg_file/one_class_v1_1800.weights"
# modelWeights = "cfg_file/yolov3-tiny.weights"
# net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
# #net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL) #AMD GPU사용할 경우, 근데 nvidia도 되던데 리눅스에선 안돌아감..


# 모델의 출력층을 호출하기 위한 함수
# YOLO v3는 스케일이 다른 3개의 출력이 존재
def getOutputsNames(net):
    # 네트웍의 모든 이름을 가져오는 함수 ,총 갯수는 254개, 실제 레이어는 100 몇개,
    layersNames = net.getLayerNames()  
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    # 실제 output 위치는 getUnconnectedOutLayers에서 1뺌
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]  


# 사각형으로 바운딩된 곳을 표시
# 프레임, 클래스ID, 클래스에 대한 일치 확률,사각형 정보등을 받음
def drawPred(frame, classId, conf, left, top, right, bottom, classes): 
    # 바운딩박스 그리기 위한 함수, frame에 사각형 좌표(left, top, right, bottom)를 기반으로 그려진다.
    # print(f"left : {left}, top : {top}, right : {right}, bottom : {bottom}")
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 6)

    label = '%.2f' % conf
    # Get the label for the class name and its confidence
    # 응급환자가 발견된 경우 라벨을 'Fainting People'로 선언
    if classes:
        assert (classId < len(classes))
        # label = '%s:%s' % (classes[classId], label)
        # label = '%s' % (classes[classId])
        label = 'Fainting People'


    # 바운딩박스가 표시된 영역 위에 라벨을 표시
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.putText(frame, label, (left, top - 15), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)


# Remove the bounding boxes with low confidence using non-maxima suppression
# Confidence threshold와 non-maxima suppression 값을 기준으로 객체를 구분하는 함수
def postprocess(frame, outs, classes, fainting_people):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    # 3개의 출력층(outs)
    for out in outs:
        # 출력층별 검출된 영역정보를 detection으로 호출
        for detection in out:
            # classfile에 지정된 클래스 수만큼의 확률값을 가져옴
            # 해당모델에서 정의된 클래스는 1이므로 1개의 확률값을 가져옴
            scores = detection[5:]  
            # 클래스별 확률에서 가장 값이 높은 인덱스를 가져옴
            classId = np.argmax(scores)  
            # 클래스별 확률에서 가장 값이 높은 인덱스에 대한 확률 값을 confidence 변수에 저장
            confidence = scores[classId]  

            # confidence 값이 confThreshold보다 더 높을 경우 객체로 간주하고 바운딩박스를 만듬
            # detection에서 0~3까지의 인덱스에는 좌표(x축 중앙, y축 중앙, 너비, 높이)정보가 있음
            # 그외(인덱스 4 이상부터)에는 객체 확률정보가 있음
            if confidence > confThreshold:  
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # NMS 알고리즘을 통해 같은 포인트에서 다수의 바운딩 박스가 생성되어 있는걸 좀더 정확하게 판별해서 최소화
    # 결과를 indices 변수에 저장 
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    
    # indices 값이 1보다 큰 경우 심정지 환자 발생으로 간주하여 아래 코드 실행
    # frame에 'Fall Detection' 라벨 처리
    if len(indices) >= 1:
        fainting_people = len(indices)
        # 프레임 중앙 위 텍스트 표시
        label = 'Fall Detection'
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 5, 5)
        print(labelSize, baseLine)
        top = max(150, labelSize[1])
        print(top)
        cv2.rectangle(frame, (frameWidth // 2 - 300, top - round(1.5 * labelSize[1])),
                      (frameWidth // 2 - 300 + labelSize[0], top + baseLine), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, label, (frameWidth // 2 - 300, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 5)

    # indice 값을 토대로 심정지 환자의 영역을 drawpred 함수를 통해 frame에 표시
    for i in indices:
        i = i[0]  # 2차원 행렬이므로
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height, classes)

    # 발견된 심정지 환자 수(fainting_people) 반환
    return fainting_people

# main.py 함수에서 50프레임 주기로 실행되는 함수
# 객체인식 진행
def yolo(frame):
    # print("YOLO Start in")
    # print(frame.shape)
    fainting_people = 0
    start_time = time.time()

    # 모델 관련 weight, config, class 호출
    # classes에는 객체에 대한 클래스 정보(fainting_people)가 있음
    classesFile = "Object_detection/cfg_file/one.names"
    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    # 모델에 대한 config 정보
    modelConfiguration = "Object_detection/cfg_file/one.cfg"
    # 모델에 대한 weight 값이 담긴 정보
    modelWeights = "Object_detection/cfg_file/one_class_v2_2800.weights"

    # 네트워크 관련 설정
    # 모델 config과 weight를 입력하여 net 에 저장
    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # 프레임 전처리 및 네트워크 Input
    # 프레임을 416x416 형식으로 전처리하여 blob에 저장
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
    # 프레임을 모델에 입력
    net.setInput(blob)
    # 입력된 결과를 outs에 저장
    outs = net.forward(getOutputsNames(net))

    # postprocess 함수에서 객체인식을 분석하고 결과를 fainting_people 변수에 저장
    # fainting_people에는 심정지 환자를 발견한 수가 저장됨
    # postprocess 함수에서 심정지 환자가 발견될 경우 해당 환자의 영역을 frame에 표시
    fainting_people = postprocess(frame, outs, classes, fainting_people)

    # cv2.imwrite(url, frame)
    print(f"YOLO inference time : {round(time.time() - start_time, 3)}")
    # frame과 fainting_people 변수를 list 형식으로 반환
    return [frame, fainting_people]

# 테스트 시
if __name__ == "__main__":  
    print("Main Process....")
    start_time = time.time()
    frame = cv2.imread('7.jpg')
    print(frame.shape)
    frame = cv2.resize(frame, (416, 416))
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
    print("Pass1")
    net.setInput(blob)
    print("Pass2")
    outs = net.forward(getOutputsNames(net))
    print("Pass3")
    postprocess(frame, outs)
    print("Pass4")
    cv2.imwrite('result.jpg', frame)
    print(f"frame shape : {frame.shape}, YOLO Process....Success")
    print(f"inference time : {round(time.time() - start_time, 4)}")

