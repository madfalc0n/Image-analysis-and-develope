from flask import Flask, Response, render_template
from multiprocessing import Process, Queue, Lock
import time
import cv2
import yolo_pro


app = Flask(__name__)
w1 = Queue()  # 프레임 넣는곳
w2 = Queue()  # 웹에 보여줄 인식결과 저장하는 곳
video = cv2.VideoCapture("swoon.mp4")

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

def generate1():
    global video
    while True:
        hasframe, frame = video.read()
        cpy_frame = frame.copy()
        cpy_frame = cv2.resize(cpy_frame, (416,416))
        if frame is None:
            continue

        # JPEG 형식으로 인코딩
        (flag, encodedImage) = cv2.imencode(".jpg", cpy_frame)

        # 인코딩 성공적으로 되었는지 확인
        if not flag:
            continue

        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')

def generate2():
    # global w2
    # while True:
    #     frame = w2.get()
    global video
    while True:
        hasframe, frame = video.read()
        if frame is None:
            continue

        #이미지 리사이즈
        cpy_frame = frame.copy()
        cpy_frame = cv2.resize(cpy_frame, (416, 416))

        # JPEG 형식으로 인코딩
        (flag, encodedImage) = cv2.imencode(".jpg", cpy_frame)

        # 인코딩 성공적으로 되었는지 확인
        if not flag:
            continue

        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


def create_frame(q):
    i = 0
    while (True):
        hasFrame, frame = video.read()
        print(f"process...index : {i}")
        if frame is -1:
            break
        if not hasFrame:
            print("process...end")
            break
        else:  # 프레임 있을경우 영상 처리 해줘야제
            # cpy_frame = cv2.resize(frame, (416, 416))
            # cv2.imshow('image', cpy_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            #time.sleep(0.03)
            if i % 150 == 0:  # 30프레임당 큐에 넣기
                q.put(frame)
                #print(f"index: {i} INPUT FRAMES")
        i += 1
    video.release()
    cv2.destroyAllWindows()


def processing_frame(q,q2):
    while True:
        frame = q.get()
        start_time = time.time()
        #print(f"OUTPUT FRAMES : {frame.shape}")
        frame = yolo_pro.yolo(frame)
        # cpy_frame = cv2.resize(frame,(416,416))
        # cv2.imshow('image2', cpy_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        #print(f"inference time : {round(time.time()-start_time, 4)}")
        q2.put(frame)
        if frame is -1:
            break

if __name__ == "__main__":
    p1 = Process(target=create_frame, args=(w1,))
    p2 = Process(target=processing_frame, args=(w1,w2))
    p1.start()
    p2.start()
    app.run(host='0.0.0.0', port=8080, debug=True, threaded=True, use_reloader=False)




