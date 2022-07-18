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
from PIL import Image

# upload img and resize if it needed for prediction
img_path = 'C:/Users/Grimm/Desktop/analiz_foto/r4.jpg'
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
fullbody_cascade = cv2.CascadeClassifier('C:/Users/Grimm/Desktop/haarcascades/haarcascade_fullbody.xml')
smile_cascade = cv2.CascadeClassifier('C:/Users/Grimm/Desktop/haarcascades/haarcascade_smile.xml')
frontal_face_cascade = cv2.CascadeClassifier('C:/Users/Grimm/Desktop/haarcascades/haarcascade_frontalcatface.xml')
eye_cascade = cv2.CascadeClassifier('C:/Users/Grimm/Desktop/haarcascades/haarcascade_lefteye_2splits.xml')
profile_face_cascade = cv2.CascadeClassifier('C:/Users/Grimm/Desktop/haarcascades/haarcascade_profileface.xml')
# persents for img resizing with correct aspect ratio
pers = 100
real_size = img.shape[0]

#get creation date
def get_date_taken(img_path):
    return print(Image.open(img_path)._getexif()[36867])

# img prepearing
def persents_counter(img):
    if img.shape[0] >= img.shape[1]:
        real_size = img.shape[0]
    else:
        real_size = img.shape[1]
    pers = 80000 / real_size

def img_resize(img, pers):
    if img.shape[0] > 800 or img.shape[1] > 800:
        width = int(img.shape[1] * pers / 100)
        height = int(img.shape[0] * pers / 100)
        dim = (width, height)
        resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    else:
        resized_img = img
    print(img.shape)
    print(resized_img.shape)
    return resized_img
# drawing and showing img for visualisation and testing

def draw_result(search_result, resized_img, name='img'):
    for (x, y, w, h) in search_result:
        cv2.rectangle(resized_img, (x, y), (x + w, y + h), (12, 150, 100), 2)
    cv2.imshow(name, resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# set methods for detection body or else on a img
def fullbody_search(resized_img):
    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    search_result = fullbody_cascade.detectMultiScale(gray, 1.1, 1, minSize=(250, 250))
    print(len(search_result)) # print number of detection objects
    return search_result

def smile_search(resized_img):
    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    search_result = smile_cascade.detectMultiScale(gray, 1.5, 11, minSize=(500, 500))
    return search_result

def frontal_face_search(resized_img):
    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    search_result = frontal_face_cascade.detectMultiScale(gray, 1.1, 2, minSize=(100, 100))
    return search_result

def eye_search_for_face(resized_img):
    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    search_result = eye_cascade.detectMultiScale(gray, 1.1, 2, minSize=(70, 70))
    return search_result

def eye_search_for_body(resized_img):
    gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    search_result = eye_cascade.detectMultiScale(gray, 1.1, 1, minSize=(10, 10), maxSize=(30,30))
    return search_result

# profile_face_search don`t worcking correct for all img yet. Need to add a border for img.
def profile_face_search(resized_img_bd):
    gray = cv2.cvtColor(resized_img_bd, cv2.COLOR_BGR2GRAY)
    search_result = profile_face_cascade.detectMultiScale(gray, 1.1, 1, minSize=(500,500))
    return search_result

def resized_img_with_border(resized_img):
    resized_img = cv2.copyMakeBorder(resized_img, 10, 10, 10, 10, 0)
    return resized_img


if img.shape[0] >= img.shape[1]:
    real_size = img.shape[0]
else:
    real_size = img.shape[1]
pers = 80000 / real_size

resized_img = img_resize(img, pers)
rseized_img = cv2.copyMakeBorder(resized_img, 10, 10, 10, 10, 0)
num_of_rotate = 0
get_date_taken(img_path)
while num_of_rotate != 4:

    if num_of_rotate != 0:
        resized_img = cv2.rotate(resized_img, cv2.ROTATE_90_CLOCKWISE)

    search_result = fullbody_search(resized_img)
    if len(search_result) == 1:
        draw_result(search_result, resized_img, name = 'fullbody')
        num_of_rotate = 0
        break

    search_result = smile_search(resized_img)
    if len(search_result) == 1:
        draw_result(search_result, resized_img, name = 'smile')
        num_of_rotate = 0
        break

    search_result = frontal_face_search(resized_img)
    if len(search_result) == 1:
        draw_result(search_result, resized_img, name='frontal face')
        num_of_rotate = 0
        break

    search_result = profile_face_search(resized_img)
    if len(search_result) == 1:
        draw_result(search_result, resized_img, name = "profile face")
        num_of_rotate = 0
        break

    else:
        if num_of_rotate >= 3:
            cv2.imshow("not detected", resized_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break
        else:
            num_of_rotate += 1
