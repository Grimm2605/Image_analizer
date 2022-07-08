"""
На вход подаётся картинка
Проверить ориентацию картинки и при необходимости развернуть её (OpenCV)
Изображение человека может быть горизонтальное и вертикальное, челюстей всегда горизонтальное
Определить контур и развернуть к началу координат
Передать картинку в нужную сеть для классификации (TF? Keras?)
Сохранить картинку в целевой деректории с верными параметрами


Задача 1: Создать или преобразовать картинку к правильному формату
Задача 2: Классифицировать картинку по профилю
Задача 2.1: Собрать трансформатор изображений для приведения к нужному формату и размеру (не актуально
у каждой клиники своя техника, формат изображения не будет универсальным)
Задача 2.2: Выделить контур изображения
Задача 2.3: Подготовить обучающие выборки для МЗ

Логика:
1 Фотография проверяется на наличие: Тела в целом, затем наличие головы, затем наличие зубов.
Если нет совподений поворот картинки на 90 градусов, повторный цикл.
2 Фотография с головой или телом проверяется на наличие глаз. Если глаза найдены, поворот таким образом,
что бы глаза находились ввеху изображения.
3 Сохранить повёрнутое изображение в директории
"""
import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt

# upload img and resize if it needed for prediction
img = cv2.imread('C:/Users/Grimm/Desktop/analiz_foto/9.jpg', cv2.IMREAD_COLOR)
fullbody_cascade = cv2.CascadeClassifier('C:/Users/Grimm/Desktop/haarcascades/haarcascade_fullbody.xml')
smile_cascade = cv2.CascadeClassifier('C:/Users/Grimm/Desktop/haarcascades/haarcascade_smile.xml')
frontal_face_cascade = cv2.CascadeClassifier('C:/Users/Grimm/Desktop/haarcascades/haarcascade_frontalcatface.xml')
eye_cascade = cv2.CascadeClassifier('C:/Users/Grimm/Desktop/haarcascades/haarcascade_lefteye_2splits.xml')
profile_face_cascade = cv2.CascadeClassifier('C:/Users/Grimm/Desktop/haarcascades/haarcascade_profileface.xml')

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

# drawing and showing img for visualisation and testing

def draw_result(search_result, resized_img):
    for (x, y, w, h) in search_result:
        cv2.rectangle(resized_img, (x, y), (x + w, y + h), (12, 150, 100), 2)
    cv2.imshow('image', resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# set methods for detection body on a img
def fullbody_search(resized_img):
    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    search_result = fullbody_cascade.detectMultiScale(gray, 1.1, 1, minSize=(100, 100))
    return search_result

def smile_search(resized_img):
    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    search_result = smile_cascade.detectMultiScale(gray, 1.5, 11, minSize=(100, 100))
    return search_result

def frontal_face(resized_img):
    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    search_result = frontal_face_cascade.detectMultiScale(gray, 1.1, 2, minSize=(100, 100))
    return search_result

def eye_search_for_face(resized_img):
    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    search_result = eye_cascade.detectMultiScale(gray, 1.1, 2, minSize=(70, 70))
    return search_result

def eye_search_for_body(resized_img):
    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    search_result = eye_cascade.detectMultiScale(gray, 1.1, 1, minSize=(10, 10), maxSize=(20,20))
    return search_result

# profile_face_search don`t worcking yet
def profile_face_search(resized_img):
    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    search_result = profile_face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(70, 70))
    return search_result


if __name__ == "__main__":
    resized_img = img_resize(img)
    search_result = profile_face_searcer(resized_img)
    draw_result(search_result, resized_img)
