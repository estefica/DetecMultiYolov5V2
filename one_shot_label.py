import cv2,os,shutil
import numpy as np
import lab_multis as lm
import imgaug as ia # imgaug
import imgaug.augmenters as iaa

from utils.datasets import letterbox
from utils.general import check_img_size
path_imagenes = '/content/prueba_data/' #'C:/Users/carit/PycharmProjects/tutorial/data_2/'#'C:/Users/carit/Documents/L377506/L377506/oblique/'#
path_save_images = '/content/random/images/'#'C:/Users/carit/PycharmProjects/tutorial/carpetadata/'#
path_save_labels = '/content/random/labels/'
new_center = 0

def crear_texto(img,objeto,w_real,h_real,name,cx,cy,boxes,prox):
    h_label =h_real/(prox*img.shape[0])
    w_label = w_real / (prox*img.shape[1])
    texto_label = '{} {} {} {} {}'.format(objeto, cx, cy, w_label, h_label)
    doc = open(path_save_labels+name+str(boxes)+'_'+str(prox)+'.txt', 'w')
    
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
              new_center = int((limx2 - limx1) / 2)
              new_center = limx2 - new_center
              limx1 = 0
              cx_txt = new_center / (limx2 - limx1)
              #print(cx_txt)
              w_real *= 0.97
                
            if limx2 > ancho:
              new_center = int((limx2 - limx1) / 2)
              limx2 = ancho
              cx_txt = new_center / (limx2 - limx1)
              w_real *= 0.97
                
            if limy1 < 0:
              new_center = int(limy2 - (limy2 - limy1) / 2)
              limy1 = 0
              cy_txt = new_center / (limy2 - limy1)
              h_real *= 0.97
                

            if limy2 > alto:
              new_center = int((limy2 - limy1)/2)
              #print(new_center)
              limy2 = alto
              #print(f'soy alto dos:{limy2 - limy1}')
              cy_txt = new_center/(limy2 - limy1)
              #print(f'\n el cy:{cy_txt}')
              h_real *= 0.97
                
            imgcrop = img[limy1:limy2, limx1:limx2]
            imgcrop = ia.imresize_single_image(imgcrop, 0.5, "cubic")
            crear_texto(img_t, objeto, w_real, h_real, name, cx_txt, cy_txt, boxes, prox)
            cv2.imwrite(path_save_images+name+str(boxes)+'_'+str(prox)+'.jpg', imgcrop)
            if prox != 1:
              img_n = cv2.imread(path_save_images+name+str(boxes)+'_'+str(prox)+'.jpg')
              img_n = cv2.resize(img_n, (960, 720))
              cv2.imwrite(path_save_images + name + '_new_' + str(boxes) + '_' + str(prox) + '.jpg',img_n)
              shutil.copy(path_save_labels+name+str(boxes)+'_'+str(prox)+'.txt',path_save_labels + name + '_new_' + str(boxes) + '_' + str(prox) + '.txt')
                       
