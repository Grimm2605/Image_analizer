# На вход подаётся картинка
# Проверить ориентацию картинки и при необходимости развернуть её (OpenCV)
## Изображение человека может быть горизонтальное и вертикальное, челюстей всегда горизонтальное
## Определить контур и развернуть к началу координат
## Найти глаза и повернуть так, что бы они были в верхней половине изображения
# Определить на картинке изображено лицо или отдельная часть (OpenCV)
# Передать картинку в нужную сеть для классификации (TF? Keras?)
# Сохранить картинку в целевой деректории с верными параметрами


# Задача 1: Создать или преобразовать картинку к правильному формату
# Задача 2: Классифицировать картинку по профилю
# Задача 2.1: Собрать трансформатор изображений для приведения к нужному формату и размеру
# Задача 2.2: Выделить контур изображения
# Задача 2.3: Подготовить обучающие выборки для МЗ
import numpy as np
import cv2
from matplotlib import pyplot as plt

#upload img and resize if it needed for prediction
img = cv2.imread('C:/Users/Grimm/Desktop/analiz_foto/0.png', cv2.IMREAD_COLOR)
if img.shape[0] > 2500 or img.shape[1] > 2500:
    scale_percent = 20 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    print(resized.shape)
elif 1500 <= img.shape[0] <= 2500 or 1500 <= img.shape[1] <= 2500:
    scale_percent = 50 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    print(resized.shape)
else:
    resized = img
    print(resized.shape)

#set a mechanism for detection body on a img
face_cascade = cv2.CascadeClassifier('C:/Users/Grimm/Desktop/haarcascades/haarcascade_fullbody.xml')
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 1, minSize = (100,100))

#drowing and showing img
for (x, y, w, h) in faces:
    cv2.rectangle(resized, (x, y), (x + w, y + h), (12, 150, 100), 2)
cv2.imshow('image', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()