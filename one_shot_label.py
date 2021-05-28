import cv2,os
import numpy as np
import lab_multis as lm

from utils.datasets import letterbox
from utils.general import check_img_size
path_imagenes = '/content/prueba_data/' #'C:/Users/carit/PycharmProjects/tutorial/data_2/'#'C:/Users/carit/Documents/L377506/L377506/oblique/'#
path_save_images = '/content/train/images/'#'C:/Users/carit/PycharmProjects/tutorial/carpetadata/'#
path_save_labels = '/content/train/labels/'
new_center = 0

def crear_texto(img,objeto,w_real,h_real,name,cx,cy,boxes,prox):
    h_label =h_real/(prox*img.shape[0])
    w_label = w_real / (prox*img.shape[1])
    texto_label = '{} {} {} {} {}'.format(objeto, cx, cy, w_label, h_label)
    doc = open(path_save_labels+name+str(boxes)+'.txt', 'w')
    doc.write(texto_label)


def one_shot_imagen(prox,img,name,labels):
    ancho1 = img.shape[1]
    alto1 = img.shape[0]
    img = cv2.resize(img, (int(ancho1 * 0.6), int(alto1 * 0.6)))  # width and height
    # CARGAR LAS DIMENSIONES
    ancho = img.shape[1]
    alto = img.shape[0]
    ##--CARGAR LABELS

    for boxes in range(labels.shape[0]):
        img_t = img
        if labels.shape[1] == 5:
            cx_real = int(labels[boxes][1] * ancho)
            cy_real = int(labels[boxes][2] * alto)
            w_real = int(labels[boxes][3] * ancho)
            h_real = int(labels[boxes][4] * alto)
            objeto = int(labels[boxes][0])
            h_mod = h_real * 1.1
            w_mod = w_real * 1.1
            h_realn = check_img_size(int(h_mod), 32)
            letterbox(img, 32)
            img_t = letterbox(img_t, (h_realn, h_realn), stride=32)[0]
            limy2 = cy_real + int(prox * (img_t.shape[0] / 2))
            limy1 = cy_real - int(prox * (img_t.shape[0] / 2))
            limx2 = cx_real + int(prox * (img_t.shape[1] / 2))
            limx1 = cx_real - int(prox * (img_t.shape[1] / 2))
            cx_txt = 0.5
            cy_txt = 0.5
            if limx1 < 0:
                new_center = int(limx1)
                new_center /= img_t.shape[1]
                cx_txt = 0.5 + new_center
                w_real *= 0.95
                limx1 = 0
            if limx2 > ancho:
                new_center = int(limx2 - ancho)
                new_center /= img_t.shape[0]
                cx_txt = 0.5 + new_center
                w_real *= 0.95
                limx2 = ancho
            if limy1 < 0:
                new_center = limy1
                new_center /= img_t.shape[1]
                cy_txt = 0.5 + new_center
                h_real *= 0.95
                limy1 = 0

            if limy2 > alto:
                new_center = int(limy2 - alto)
                new_center /= img_t.shape[0]
                cy_txt = 0.5 + new_center
                h_real *= 0.95
                limy2 = alto
            imgcrop = img[limy1:limy2, limx1:limx2]
            crear_texto(img_t, objeto, w_real, h_real, name, cx_txt, cy_txt, boxes, prox)
            cv2.imwrite(path_save_images + name + str(boxes) + '.jpg', imgcrop)
