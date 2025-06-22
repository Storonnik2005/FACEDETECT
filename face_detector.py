
import cv2
import dlib

# Загрузка предобученной модели Dlib для детекции лиц
detector = dlib.get_frontal_face_detector()

# Загрузка предобученной модели Dlib для определения ключевых точек лица (landmarks)
# Вам нужно будет скачать файл shape_predictor_68_face_landmarks.dat
# и указать к нему путь.
# Например: predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Открытие веб-камеры
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Ошибка: Не удалось открыть камеру.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Ошибка: Не удалось получить кадр.")
        break

    # Преобразование кадра в оттенки серого для Dlib
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Обнаружение лиц на кадре
    faces = detector(gray)

    for face in faces:
        # Рисование прямоугольника вокруг лица
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Определение ключевых точек лица
        landmarks = predictor(gray, face)

        # Рисование ключевых точек
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    # Отображение кадра с обнаруженными лицами
    cv2.imshow('Face Detection', frame)

    # Выход по нажатию 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()


