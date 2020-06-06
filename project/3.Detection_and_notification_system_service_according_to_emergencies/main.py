from flask import Flask, Response, render_template
from multiprocessing import Process, Queue, Lock
import time
import cv2
from Object_detection import yolo  # 욜로 사용시
# from Pose_estimation import openpose,openpose_single #오픈포즈 사용시
from ESMS import send_sms
from datetime import datetime
import argparse

# Flask 웹서버 실행, __name__ 실행
app = Flask(__name__)
print("Queue waiting")

# url 경로 '/' 호출 시 templates/index.html로 렌더링
@app.route('/')
def index():
    return render_template('/index.html')

# url 경로 '/view_info1' 호출 시 원본영상을 프레임별로 반환
@app.route('/view_info1')
def view_info1():
    return Response(generate1(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# url 경로 '/view_info2' 호출 시 객체인식 된 결과를 프레임별로 반환
@app.route('/view_info2')
def view_info2():
    return Response(generate2(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# mimetype="multipart/x-mixed-replace; boundary=frame"
# 서버가 클라이언트에게 보내주는 HTTP 헤더타입
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
        # 응급환자 발생 수를 저장한 배열(Queue) 호출
        global fainting_count
        # Queue(w1)에 저장된 Frame 호출
        frame = q.get()  

        # main.py 실행시 -m 'yolo' 입력한 경우 ,또는 옵션을 주지 않은 경우 Yolo 진행
        if mode == 'yolo':
            #Object_detection/yolo.py의 yolo 함수 호출, 객체인식 진행
            #yolo 함수에서 처리된 결과를 result 변수에 저장, list 형식으로 [인식된영상, 응급환자 수] 반환
            result = yolo.yolo(frame)

        # main.py 실행시 -m 'openpose' 입력한 경우 ,openpose 모드로 진행, 사용x
        # elif mode == 'openpose':
        # result = openpose.openpose(frame)
        # result = openpose_single.openpose(frame)

        # main.py 실행시 -m 옵션을 통해 'yolo' 또는 'openpose' 외 다른 명령을 입력한 경우 에러로 처리, break
        else:
            print("Error You choice '-m [yolo] or [openpose]'")
            break
        
        # 객체인식 후 바운딩박스 처리된 영상을 frame 변수에 저장 
        frame = result[0]
        # 객체인식 후 응급환자 발생 수를 fainting_people 변수에 저장
        fainting_people = result[1]

        # fainting_people 값이 0보다 클 경우 응급환자 발생으로 간주하여 Queue 처리
        # 제일 우선적으로 들어온 데이터(0번 인덱스)를 제거하고 마지막 항목에 새로운 데이터(1)를 추가
        if fainting_people > 0:
            print("find emergency patients")
            fainting_count.pop(0)
            fainting_count.append(1)
        
        # fainting_people 값이 0과 같거나 작은경우 응급환자가 발생하지 않은것으로 간주하여 Queue 처리
        # 제일 우선적으로 들어온 데이터(0번 인덱스)를 제거하고 마지막 항목에 새로운 데이터(0)를 추가
        else:
            fainting_count.pop(0)
            fainting_count.append(0)
        print(f"fainting_count : {fainting_count}")
        print(f"fainting_count : {sum(fainting_count)}")

        # 리스트 'fainting_count'에 존재하는 값들의 합이 10일 경우 응급상황으로 간주하여 SMS 전송함수 실행
        if sum(fainting_count) == 10:
            #print("This is emergency scenario. send SMS")
            
            # send_sms 함수를 호출하여 관리자에게 문자메시지 전송
            send_esms()
            
            # SMS 전송 후 fainting_count 초기화
            fainting_count = [0] * 10
            print(f"fainting_count init : {fainting_count}")

        # print(f"inference time : {round(time.time()-start_time, 4)}")
        
        # q2(w3)에 객체인식 결과 프레임을 input
        q2.put(frame)
        #frame 없는경우 종료
        if frame is -1:
            break

# Naver Cloud Platform에서 제공하는 SMS API 서비스를 이용하여 SMS문자를 보내기 위한 함수
def send_esms():
    print("This is emergency scenario. Send SMS")
    # 오늘 날짜 정보를 today 변수에 저장
    today = time.strftime('%Y년 %m월 %d일', time.localtime(time.time()))
    # 현재 시간을 cur_time 변수에 저장
    cur_time = time.strftime('%H시 %M분 %S초', time.localtime(time.time()))
    # 오늘 날짜와 현재 시간을 today_cur_time 변수에 저장
    today_cur_time = today + '\n' + cur_time
    # 보내는 사람의 번호를 str형식으로 send_phonenum 변수에 저장
    send_phonenum = 'yourphone'  # '010-1234-5678'
    # 받는 사람의 번호를 list(str) 형식으로 recv_phonenum 변수에 저장
    recv_phonenum = ['sendphone']  # 1명에게 보낼경우 ['010-1234-5678'] 2명 이상에게 보낼 경우 ['010-1234-5678','010-3333-7777']
    
    # ESMS/send_sms.py의 send_sms 함수에 보내는 사람, 받는 사람의 전화번호와 날짜와시간에 대한 정보를 넘겨줌
    # SMS 발송 후 응답결과(json 포멧)를 result 변수에 저장 
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

