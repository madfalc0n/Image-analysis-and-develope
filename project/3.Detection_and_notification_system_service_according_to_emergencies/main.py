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


# 원본 영상을 송출하기 위한 함수
def generate1():
    #init시 생성된 Queue 변수 호출 
    global w2
    while True:
        # Queue에 저장된 원본 프레임 호출
        frame = w2.get()  

        # Queue에 저장된 프레임이 없는 경우 continue
        if frame is None:
            continue

        #원본 frame 복사 후 640x640 으로 리사이즈
        cpy_frame = frame.copy()
        cpy_frame = cv2.resize(cpy_frame, (640, 640))

        # JPEG 형식으로 인코딩
        (flag, encodedImage) = cv2.imencode(".jpg", cpy_frame)

        # 인코딩이 성공적으로 되었는지 확인
        if not flag:
            continue

        # yield the output frame in the byte format
        # 매 프레임마다 str - > byte 포멧 형식으로 변환 후 yield를 통해 반환
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')

# 인식결과에 대한 영상을 송출하기 위한 함수
def generate2():
    #init시 생성된 Queue 변수 호출
    global w3
    while True:
        # 인식결과 프레임이 저장된 Queue 호출
        frame = w3.get()  
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

# 프레임 생성 ,q = w1 ,q2 = w2
def create_frame(input_image, q, q2):
    # 영상파일 호출, 실시간일 경우 0
    video = cv2.VideoCapture(input_image)
    i = 0
    # Loop
    while True:
        # read함수를 통해 프레임 호출
        hasFrame, frame = video.read()
        # 프레임을 q2(원본영상을 웹에 송출하기위한 Queue, w2)에 input
        q2.put(frame)

        #영상이 끝난경우 종료
        if frame is -1:
            break
        
        #프레임이 없는경우 종료
        if not hasFrame:
            print("process...end")
            break

        # 프레임 있을경우 영상 처리
        else: 
            #딜레이
            time.sleep(0.03)

            # 50프레임(약1.5초 주기)당 실행
            if i % 50 == 0:  
                # 응급환자 발생여부를 판독하기 위한 q(w1)에 input
                q.put(frame)  
                print("------------------------------------")
                print(f"Process {i} FRAMES")
        i += 1
    # 종료 후 객체 close
    video.release()
    cv2.destroyAllWindows()



# frame 처리 ,q = w1 ,q2 = w3
# w1에 프레임이 있는 경우에만 처리됨
# 응급환자 발생여부를 판독하기 위한 프레임이 있을경우 처리
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
    #main.py 실행시 옵션을 설정할 수 있도록 argparse 객체 지정 
    parser = argparse.ArgumentParser()

    #main.py 실행시 추가적으로 줄 수 있는 인자값 설정
    #-i 옵션을 통해 image 또는 video 호출 가능, default 값으로 'videos/20200424_1.mp4'가 지정됨
    parser.add_argument('-i', '--image', type=str, default='videos/20200424_1.mp4', help='input image or video')
    #-m 옵션을 통해 객체인식 모드를 지정가능, yolo 모드와 openpose 모드를 지정할 수 있으며 default 값으로 'yolo'가 지정됨
    parser.add_argument('-m', '--mode', type=str, default='yolo', help='choice "yolo" or "openpose"')
    #인자값을 사용하기 위한 객체 지정
    args = parser.parse_args()
    #이미지 관련 인자값에 대해 저장
    input_image = args.image
    #인식관련 인자값에 대해 저장
    mode = args.mode

    
    print("Main status")

    # 프로세스 간 데이터를 주고받기 위한 큐 할당
    w1 = Queue()  
    w2 = Queue()
    w3 = Queue()
    print("Queue Ready")

    #응급환자 발생 수를 저장하기 위한 배열(Queue) 설정
    fainting_count = [0] * 10
    
    # 각 프로세스별로 함수 작동하도록 설정(병렬처리)
    p1 = Process(target=create_frame, args=(input_image, w1, w2))  
    p2 = Process(target=processing_frame, args=(mode, w1, w3))

    #프로세스 가동
    p1.start()
    print("process 1 start")
    p2.start()
    print("process 2 start")
    print("Process Ready")

    #플라스크 웹 서버 실행
    app.run(host='0.0.0.0', port=8992, debug=True, threaded=True, use_reloader=False)
    print("app start")

