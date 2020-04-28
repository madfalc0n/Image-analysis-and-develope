import cv2
import argparse
import numpy as np
import os.path
import os
import time

# 파라미터 초기화
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.3  # Non-maximum suppression threshold
inpWidth = 416  # Width of network's input image
inpHeight = 416  # Height of network's input image


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


# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()  # 네트웍의 모든 이름을 가져오는 함수 ,총 갯수는 254개, 실제 레이어는 100 몇개,
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]  # 실제 output 위치는 getUnconnectedOutLayers에서 1뺌


# Draw the predicted bounding box
# 사각형으로 바운딩한 곳을 표시
def drawPred(frame, classId, conf, left, top, right, bottom, classes):  # 클래스ID, 클래스에 대한 일치 확률,사각형 정보
    # 바운딩박스 그리는 함수
    # print(f"left : {left}, top : {top}, right : {right}, bottom : {bottom}")
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 6)

    label = '%.2f' % conf
    # Get the label for the class name and its confidence
    if classes:
        assert (classId < len(classes))
        # label = '%s:%s' % (classes[classId], label)
        # label = '%s' % (classes[classId])
        label = 'Fainting People'

        # Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    # 바운딩박스 위 해당 클래스 표시
    cv2.putText(frame, label, (left, top - 15), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)


# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs, classes, fainting_people):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]  # 확률 80개의 값을 가져옴
            classId = np.argmax(scores)  # 확률에서 높은거 가져옴
            confidence = scores[classId]  # 확률 값 가져옴
            if confidence > confThreshold:  # confThreshold보다 더 높을 경우, 박스를 만듬
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    # 같은 포인트에서 다수의 바운딩 박스가 생성되어 있는걸 좀더 정확하게 판별해서 최소화 해줌
    # print(f"indices values : {indices} ,  {len(indices)}")
    if len(indices) >= 1:
        fainting_people = len(indices)
        # 프레임 중앙 위 텍스트
        label = 'Fall Detection'
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 5, 5)
        print(labelSize, baseLine)
        top = max(150, labelSize[1])
        print(top)
        cv2.rectangle(frame, (frameWidth // 2 - 300, top - round(1.5 * labelSize[1])),
                      (frameWidth // 2 - 300 + labelSize[0], top + baseLine), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, label, (frameWidth // 2 - 300, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 5)

    for i in indices:
        i = i[0]  # 2차원 행렬이므로
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height, classes)

    return fainting_people


def yolo(frame):
    # print("YOLO Start in")
    # print(frame.shape)
    fainting_people = 0
    start_time = time.time()

    # 모델 관련 weight, config, class 호출
    classesFile = "Object_detection/cfg_file/one.names"
    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    modelConfiguration = "Object_detection/cfg_file/one.cfg"
    modelWeights = "Object_detection/cfg_file/one_class_v2_2800.weights"

    # 네트워크 관련 설정
    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # 프레임 전처리 및 네트워크 Input
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    outs = net.forward(getOutputsNames(net))
    fainting_people = postprocess(frame, outs, classes, fainting_people)

    # cv2.imwrite(url, frame)
    print(f"YOLO inference time : {round(time.time() - start_time, 3)}")
    return [frame, fainting_people]


if __name__ == "__main__":  # 테스트 시
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

