from flask import Flask, Response, render_template
from multiprocessing import Process, Queue, Lock
import time
import cv2
from Object_detection import yolo  # 욜로 사용시
# from Pose_estimation import openpose,openpose_single #오픈포즈 사용시
from ESMS import send_sms
from datetime import datetime
import argparse

app = Flask(__name__)
print("Queue waiting")


@app.route('/')
def index():
    return render_template('/index.html')


@app.route('/view_info1')
def view_info1():
    return Response(generate1(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route('/view_info2')
def view_info2():
    return Response(generate2(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# mimetype="multipart/x-mixed-replace; boundary=frame"
# 서버가 클라이언트에게 보내주는 HTTP 헤더로
# 브라우져 화면에 나타나는 내용이 지속적으로 변하게 가능


def generate1():  # 원본 영상을 송출하기 위한 함수
    global w2
    while True:
        frame = w2.get()  # Queue에 저장된 원본 프레임 호출

        if frame is None:
            continue
        cpy_frame = frame.copy()
        cpy_frame = cv2.resize(cpy_frame, (640, 640))

        # JPEG 형식으로 인코딩
        (flag, encodedImage) = cv2.imencode(".jpg", cpy_frame)

        # 인코딩 성공적으로 되었는지 확인
        if not flag:
            continue
        # yield the output frame in the byte format
        # byte 타입의 str형식으로 처리되는 프레임마다 반환
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


def generate2():  # 인식결과에 대한 영상을 송출하기 위한 함수
    global w3
    while True:
        frame = w3.get()  # Queue에 저장된 인식결과 프레임 호출
        if frame is None:
            continue

        # 이미지 리사이즈
        cpy_frame = frame.copy()
        cpy_frame = cv2.resize(cpy_frame, (640, 640))

        # JPEG 형식으로 인코딩
        (flag, encodedImage) = cv2.imencode(".jpg", cpy_frame)

        # 인코딩 성공적으로 되었는지 확인
        if not flag:
            continue
        # yield the output frame in the byte format
        # byte 타입의 str형식으로 처리되는 프레임마다 반환
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


def create_frame(input_image, q, q2):
    video = cv2.VideoCapture(input_image)
    i = 0
    while True:
        hasFrame, frame = video.read()
        q2.put(frame)  # 원본영상 웹 송출용 Queue
        # print(f"process...index : {i}")
        if frame is -1:
            break
        if not hasFrame:
            print("process...end")
            break
        else:  # 프레임 있을경우 영상 처리
            time.sleep(0.03)
            if i % 50 == 0:  # 50프레임당 Queue input
                q.put(frame)  # 응급환자 발생여부를 위한 Queue
                print("------------------------------------")
                print(f"Process {i} FRAMES")
        i += 1
    video.release()
    cv2.destroyAllWindows()


def processing_frame(mode, q, q2):
    while True:
        global fainting_count
        frame = q.get()  # Queue에 저장된 Frame 호출
        # start_time = time.time()
        # print(f"OUTPUT FRAMES : {frame.shape}")
        if mode == 'yolo':
            result = yolo.yolo(frame)
        # elif mode == 'openpose':
        # result = openpose.openpose(frame)
        # result = openpose_single.openpose(frame)
        else:
            print("Error You choice '-m [yolo] or [openpose]'")
            break

        frame = result[0]
        fainting_people = result[1]

        if fainting_people > 0:
            print("find emergency patients")
            fainting_count.pop(0)
            fainting_count.append(1)
        else:
            fainting_count.pop(0)
            fainting_count.append(0)

        print(f"fainting_count : {fainting_count}")
        print(f"fainting_count : {sum(fainting_count)}")
        if sum(fainting_count) == 10:
            #print("This is emergency scenario. send SMS")
            send_esms()
            fainting_count = [0] * 10
            print(f"fainting_count init : {fainting_count}")

        # print(f"inference time : {round(time.time()-start_time, 4)}")
        q2.put(frame)
        if frame is -1:
            break


def send_esms():
    print("This is emergency scenario. Send SMS")
    today = time.strftime('%Y년 %m월 %d일', time.localtime(time.time()))
    cur_time = time.strftime('%H시 %M분 %S초', time.localtime(time.time()))
    today_cur_time = today + '\n' + cur_time
    send_phonenum = 'yourphone'  # 보내는사람 번호
    recv_phonenum = ['sendphone']  # 받는사람 번호 리스트형식으로 작성
    result = send_sms.send_msg(send_phonenum, recv_phonenum, today_cur_time)
    print(f"result : {result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', type=str, default='videos/20200424_1.mp4', help='input image or video')
    parser.add_argument('-m', '--mode', type=str, default='yolo', help='choice "yolo" or "openpose"')
    args = parser.parse_args()
    input_image = args.image
    mode = args.mode

    print("Main status")
    w1 = Queue()  # 프로세스 간 데이터를 주고받기 위한 큐 할당
    w2 = Queue()
    w3 = Queue()
    fainting_count = [0] * 10
    print("Queue Ready")
    p1 = Process(target=create_frame, args=(input_image, w1, w2))  # 각 프로세스에 해당 함수 작동하도록 설정
    p2 = Process(target=processing_frame, args=(mode, w1, w3))
    p1.start()
    print("process 1 start")
    p2.start()
    print("process 2 start")
    print("Process Ready")
    app.run(host='0.0.0.0', port=8992, debug=True, threaded=True, use_reloader=False)
    print("app start")

