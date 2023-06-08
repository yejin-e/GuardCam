import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
import smtplib  # SMTP 사용을 위한 모듈
import re  # Regular Expression을 활용하기 위한 모듈
from email.mime.multipart import MIMEMultipart  # 메일의 Data 영역의 메시지를 만드는 모듈
from email.mime.text import MIMEText  # 메일의 본문 내용을 만드는 모듈
from email.mime.image import MIMEImage  # 메일의 이미지 파일을 base64 형식으로 변환하기 위한 모듈

prototxt = "SSD_data/deploy.prototxt"
caffemodel = "SSD_data/res10_300x300_ssd_iter_140000_fp16.caffemodel"
detector = cv2.dnn.readNet(prototxt, caffemodel)
num = 0  # 저장하는 사진 번호
cnt = 64
outsider = 0
buzzer_pin = 18

def ssd(image):
    global num
    (h, w) = image.shape[:2]
    target_size = (300, 300)
    input_image = cv2.resize(image, target_size)

    imageBlob = cv2.dnn.blobFromImage(input_image)
    detector.setInput(imageBlob)

    detections = detector.forward()

    results = detections[0][0]
    threshold = 0.8

    for i in range(0, results.shape[0]):
        conf = results[i, 2]
        if conf < threshold:
            continue

        box = results[i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype('int')

        cv2.putText(image, str(conf), (startX, startY - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

        cv2.imshow("image", image)

        resized_img = image[startY:endY, startX:endX]

        num += 1
        if resized_img.any():
            cv2.imwrite('test/captured%d.jpg' % num, resized_img,
                        params=[cv2.IMWRITE_JPEG_QUALITY, 100])  # 지금 촬영한 사진 저장
        return image

def processing(i, t):  # 영상 전처리
    if t == 0:  # train
        frame = 'inside/%d.jpg' % i
    elif t == 1:  # test
        frame = 'test/captured%d.jpg' % num
    else:
        return print('Type failed!')

    image = cv2.imread(frame, cv2.IMREAD_COLOR)

    if image is None:
        return print(i, 'Image load failed!')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(gray, (150, 120))
    return image

def convert(num, t):  # 영상 전처리 및 행렬 변환
    x = np.zeros((18000, 0), dtype='float32')
    if t == 0:  # train
        for i in range(num):
            image = processing(i, 0)
            image = image.reshape(-1, 1)
            x = np.concatenate((x, image), axis=1)
    else:  # test
        image = processing(num, 1)
        image = image.reshape(-1, 1)
        x = np.concatenate((x, image), axis=1)
    return x

def distance(train_feature, test_feature):
    min_dist = 10000000
    min_index = -1
    for i in range(cnt):
        dist = np.linalg.norm(train_feature[:, i:i + 1] - test_feature)
        if dist < min_dist:
            min_dist = dist
            min_index = i
    return min_dist, min_index

def test(n, m, train_feature, vec_t, capture):  # n = image
    global outsider
    value = 2000  # 외부인 여부 임계값

    y = convert(n, 1)
    y_subtraction = y - m
    test_feature = np.dot(vec_t, y_subtraction)
    min_dist, min_index = distance(train_feature, test_feature)
    print("- Similar Train Image Number:", min_index, ", min_dist: ", min_dist)

    # 일치 여부 확인하기
    if min_dist > value:
        print("외부인입니다.")
        outsider = outsider + 1

        if outsider == 7:
            outsider = 0
            play_school_bell()
            send_email(num)
            cv2.destroyAllWindows()
            capture.release()  # 카메라 연결 해제
    else:
        print("내부인입니다")
        servoMotor(16, 8, 1)  # 신호선을 16번 핀에 연결, 8의 각도로 1초동안 실행

    cv2.waitKey(10)

def cam_detect():
    print("실행시 시간이 걸릴 수 있습니다. 잠시만 기다려주세요.")

    capture = cv2.VideoCapture(0)
    if capture.isOpened() == False:
        raise Exception("카메라 연결 안됨")

    while True:
        ret, frame = capture.read()  # 카메라 영상 받기
        if not ret: break

        image = ssd(frame)  # SSD로 얼굴 검출 후 저장

        test(image, m, train_feature_vec, vec_t, capture)  # 테스트

        if cv2.waitKey(30) >= 0: break
        
    return

def mean_image(x):  # 평균 영상 구하기
    a = np.zeros((18000, 0), dtype='float32')
    m = x.mean(axis=1).reshape(-1, 1)

    for i in range(310):
        x_i = x[:, i:i + 1]
        a = np.concatenate((a, x_i - m), axis=1)
    return a, m

def eig_val_vec(a):  # 특징값, 고유 벡터 구하기
    vec = np.zeros((18000, 0), dtype='float32')
    v = -1
    cov = np.dot(a.transpose(), a)
    eig_val, eig_vec_c = np.linalg.eig(cov)
    eig_val_sum = np.sum(eig_val)
    k = 0.92

    for i in range(cnt):
        if eig_val_sum * k <= np.sum(eig_val[:i + 1]):
            v = i + 1
            break

    for i in range(v):
        eig_vec_i = np.dot(a, eig_vec_c[:, i:i + 1])
        eig_vec_i = eig_vec_i / np.linalg.norm(eig_vec_i)
        vec = np.concatenate((vec, eig_vec_i), axis=1)

    train_feature = np.zeros((v, 0), dtype='float32')
    vec_t = vec.transpose()

    for i in range(cnt):
        feature_value = np.dot(vec_t, a[:, i:i + 1])
        train_feature = np.concatenate((train_feature, feature_value), axis=1)

    return train_feature, vec_t

def train():  # 학습
    arr = convert(cnt, 0)
    a, m = mean_image(arr)
    train_feature, vec_t = eig_val_vec(a)
    return m, train_feature, vec_t

def servoMotor(pin, degree, t):
    GPIO.setmode(GPIO.BOARD)  # 핀의 번호를 보드 기준으로 설정, BCM은 GPIO 번호로 호출함
    GPIO.setup(pin, GPIO.OUT)  # GPIO 통신할 핀 설정
    pwm = GPIO.PWM(pin, 50)  # 서보모터는 PWM을 이용해야됨. 16번핀을 50Hz 주기로 설정

    pwm.start(3)  # 초기 시작값, 반드시 입력해야됨
    time.sleep(t)  # 초기 시작값으로 이동하고 싶지 않으면 이 라인을 삭제하면 된다.

    pwm.ChangeDutyCycle(12)  # 보통 2~12 사이의 값을 입력하면됨
    time.sleep(t)  # 서보모터가 이동할만큼의 충분한 시간을 입력. 너무 작은 값을 입력하면 이동하다가 멈춤

    pwm.ChangeDutyCycle(3)  # 보통 2~12 사이의 값을 입력하면됨
    time.sleep(t)  # 서보모터가 이동할만큼의 충분한 시간을 입력. 너무 작은 값을 입력하면 이동하다가 멈춤

    # 아래 두줄로 깨끗하게 정리해줘야 다음번 실행할때 런타임 에러가 안남
    pwm.stop()
    GPIO.cleanup()

def play_school_bell():
    global buzzer_pin
    # 학교종 소리를 위한 PWM 설정
    # GPIO 초기화
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(buzzer_pin, GPIO.OUT)

    pwm = GPIO.PWM(buzzer_pin, 50)  # 주파수 500Hz로 설정
    
    s = 5
    try:
        while s >= 0:
            # 학교종 소리 출력
            pwm.start(50)  # Duty Cycle 50%로 설정 (소리 크기 조절)
            time.sleep(0.5)  # 0.5초 동안 소리 유지
            pwm.stop()
            time.sleep(0.5)  # 0.5초 동안 소리 없음
            s -= 1
            

    except KeyboardInterrupt:
        pass

    finally:
        # GPIO 리소스 정리
        GPIO.cleanup()

def sendEmail(to_mail, smtp, my_account, msg):
    reg = "^[a-zA-Z0-9.+_-]+@[a-zA-Z0-9]+.[a-zA-Z]{2,3}$"  # 유효성 검사를 위한 정규표현식
    if re.match(reg, my_account):
        smtp.sendmail(my_account, to_mail, msg.as_string())
        print("정상적으로 메일이 발송되었습니다.")
    else:
        print("받으실 메일 주소를 정확히 입력하십시오.")

def send_email(idx):
    # smpt 서버와 연결
    gmail_smtp = "smtp.gmail.com"  # gmail smtp 주소
    gmail_port = 465  # gmail smtp 포트번호. 고정(변경 불가)
    smtp = smtplib.SMTP_SSL(gmail_smtp, gmail_port)
    
    # 로그인
    my_account = "deueightoh@gmail.com"
    my_password = "kajahiuowhpdhwvc"
    smtp.login(my_account, my_password)
    
    # 메일을 받을 계정
    to_mail = "hdy1558@gmail.com"
    
    # 메일 기본 정보 설정
    msg = MIMEMultipart()
    msg["Subject"] = "첨부 파일 확인 바랍니다"  # 메일 제목
    msg["From"] = my_account
    msg["To"] = to_mail
    
    # 메일 본문 내용
    content = "안녕하세요. \n\n\
    데이터를 전달드립니다.\n\n\
    감사합니다\n\n\
    "
    content_part = MIMEText(content, "plain")
    msg.attach(content_part)
    
    # 이미지 파일 추가
    image_name = "test/captured%d.jpg" % idx
    with open(image_name, 'rb') as file:
        img = MIMEImage(file.read())
        img.add_header('Content-Disposition', 'attachment', filename=image_name)
        msg.attach(img)
    
    # 받는 메일 유효성 검사 거친 후 메일 전송
    sendEmail(to_mail, smtp, my_account, msg)
    
    # smtp 서버 연결 해제
    smtp.quit()

m, train_feature_vec, vec_t = train()  # 학습
image = cam_detect()  # 캠 실행
