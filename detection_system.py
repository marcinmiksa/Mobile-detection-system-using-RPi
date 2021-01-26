import cv2
import numpy as np

# zwraca domyślną kamerę
capture = cv2.VideoCapture(0)
# ustawienia
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
capture.set(cv2.CAP_PROP_BRIGHTNESS, 60)
print("brightness", capture.get(cv2.CAP_PROP_BRIGHTNESS))

if not capture.isOpened():
    print('Kamera nie jest podlaczona')
    exit()

while True:
    # ret - zwraca wartość bool jeśli ramka jest dostępna
    ret, frame = capture.read()
    if not ret:
        print('Brak obrazu')
        exit()
    # tworzymy kopie ramki by nie pracować na "oryginalnej"
    contour_frame = frame.copy()
    # konwersja między RGB a skalą szarości
    frame_gray = cv2.cvtColor(contour_frame, cv2.COLOR_BGR2GRAY)
    # filtr do usuwania szumów
    # blur = cv2.blur(frame_gray, (5, 5))
    blur = cv2.GaussianBlur(frame_gray, (5, 5), 0)
    # trzeba przerobić na obraz binarny

    _threshold = cv2.threshold(blur, 177, 255, cv2.THRESH_BINARY)[1]
    # cv.CHAIN_APPROX_SIMPLE -usuwa zbędne punkty, oszczędzając pamięć
    contours, hierarchy = cv2.findContours(_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # print(contours)
    for contour in contours:
        area = cv2.contourArea(contour)
        # dokładność przy więcej niż 0.02 może źle rozpoznawać kształt
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        # x, y = cv2.boundingRect(contour)
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
                    # convert the (x, y) coordinates and radius of the circles to integers
                    circles = np.round(circles[0, :]).astype("int")
                    # loop over the (x, y) coordinates and radius of the circles
                    for (x, y, r) in circles:
                        # draw the circle in the output image, then draw a rectangle
                        # corresponding to the center of the circle
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
            # boundingRect() - rysuje prostokąt wokół obrazu binarnego
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

    cv2.imshow("system detection", np.hstack([color_frame, contour_frame]))
    # cv2.imshow('red mask', red_mask)
    # cv2.imshow('blue mask', blue_mask)
    # cv2.imshow('green mask', green_mask)
    cv2.imshow('threshold', _threshold)
    # cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
