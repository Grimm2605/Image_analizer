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
import cv2 as cv2
from matplotlib import pyplot as plt

# upload img and resize if it needed for prediction
img = cv2.imread('C:/Users/Grimm/Desktop/analiz_foto/0.png', cv2.IMREAD_COLOR)
fullbody_cascade = cv2.CascadeClassifier('C:/Users/Grimm/Desktop/haarcascades/haarcascade_fullbody.xml')


def img_resize(img):
    if img.shape[0] > 2500 or img.shape[1] > 2500:
        scale_percent = 20  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    elif 1500 <= img.shape[0] <= 2500 or 1500 <= img.shape[1] <= 2500:
        scale_percent = 50  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    else:
        resized_img = img
    return resized_img


# set a mechanism for detection body on a img
def fullbody_search(resized_img):
    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    search_result = fullbody_cascade.detectMultiScale(gray, 1.1, 1, minSize=(100, 100))
    return search_result


# drawing and showing img

def draw_result(search_result, resized_img):
    for (x, y, w, h) in search_result:
        cv2.rectangle(resized_img, (x, y), (x + w, y + h), (12, 150, 100), 2)
    cv2.imshow('image', resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


resized_img = img_resize(img)
search_result = fullbody_search(resized_img)
draw_result(search_result, resized_img)
