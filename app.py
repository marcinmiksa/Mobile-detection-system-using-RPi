import cv2
import numpy as np
import configure_file
from flask import Flask, render_template, Response

app = Flask(__name__)
camera = cv2.VideoCapture(configure_file.CAMERA_PATH)
# ustawienia
camera.set(cv2.CAP_PROP_FRAME_WIDTH, configure_file.FRAME_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, configure_file.FRAME_HEIGHT)
camera.set(cv2.CAP_PROP_BRIGHTNESS, configure_file.CAMERA_BRIGHTNESS)


def gen_frames():
    while True:
        # ret zwraca wartosc bool jesli ramka jest dostepna
        ret, frame = camera.read()
        if not ret:
            print('Brak obrazu')
            exit()
        # tworzymy kopie ramki
        contour_frame = frame.copy()
        # konwersja miedzy RGB a skala szarosci
        frame_gray = cv2.cvtColor(contour_frame, cv2.COLOR_BGR2GRAY)
        # filtr do usuwania szumow
        blur = cv2.GaussianBlur(frame_gray, (5, 5), 0)
        _threshold = cv2.threshold(blur, 177, 255, cv2.THRESH_BINARY)[1]
        contours, hierarchy = cv2.findContours(_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # print(contours)
        for contour in contours:
            area = cv2.contourArea(contour)
            # dokladnosc, przy wiecej niz 0.02 zle rozpoznanie ksztalty
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            x = approx.ravel()[0]
            y = approx.ravel()[1]
            # print(len(approx))
            if area > 1000:
                if len(approx) == 3:
                    cv2.drawContours(contour_frame, [approx], 0, (0, 0, 0), 2)
                    cv2.putText(contour_frame, "TROJKAT", (x, y), cv2.FONT_ITALIC, 1, (0, 0, 0))
                elif len(approx) == 4:
                    cv2.drawContours(contour_frame, [approx], 0, (0, 0, 0), 2)
                    cv2.putText(contour_frame, "PROSTOKAT", (x, y), cv2.FONT_ITALIC, 1, (0, 0, 0))
                elif len(approx) >= 10:
                    circles = cv2.HoughCircles(frame_gray, cv2.HOUGH_GRADIENT, 1.2, 100)
                    if circles is not None:
                        circles = np.round(circles[0, :]).astype("int")
                        for (x, y, r) in circles:
                            cv2.circle(contour_frame, (x, y), r, (0, 0, 0), 2)
                            cv2.putText(contour_frame, "OKRAG", (x, y), cv2.FONT_ITALIC, 1, (0, 0, 0))

        color_frame = frame.copy()

        hsv_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)

        red_mask1 = cv2.inRange(hsv_frame, (0, 90, 100), (10, 255, 255))
        red_mask2 = cv2.inRange(hsv_frame, (170, 90, 100), (179, 255, 255))
        red_mask = red_mask1 + red_mask2
        green_mask = cv2.inRange(hsv_frame, (25, 50, 70), (89, 255, 255))
        blue_mask = cv2.inRange(hsv_frame, (90, 80, 0), (120, 255, 255))

        contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                color_frame = cv2.rectangle(color_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(color_frame, 'R', (x, y), cv2.FONT_ITALIC, 1.0, (0, 0, 255))

        contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                color_frame = cv2.rectangle(color_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(color_frame, 'G', (x, y), cv2.FONT_ITALIC, 1.0, (0, 255, 0))

        contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                color_frame = cv2.rectangle(color_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(color_frame, 'B', (x, y), cv2.FONT_ITALIC, 1.0, (255, 0, 0))

        detection_frame = np.hstack([color_frame, contour_frame])
        ret, buffer = cv2.imencode('.jpg', detection_frame)
        detection_frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + detection_frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/stream')
def stream():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host=configure_file.HOST, port=configure_file.PORT, debug=configure_file.DEBUG)

# camera.release()
# cv2.destroyAllWindows()
