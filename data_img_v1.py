import os, shutil,cv2
import numpy as np

path_save_images = '/content/random/images/'#'C:/Users/carit/PycharmProjects/tutorial/carpetadata/'#
path_save_labels = '/content/random/labels/'
path_images =  '/content/prueba_data/'
lista_read = os.listdir(path_images)

def center_dimensions(labels,ancho,alto):
    cx_real = int(labels[1] * ancho)
    cy_real = int(labels[2] * alto)
    w_real = int(labels[3] * ancho)
    h_real = int(labels[4] * alto)
    objeto = int(labels[0])
    return cx_real,cy_real,w_real,h_real,objeto

def borders(cx,cy,w,h):
    px = int(cx - w / 2)
    py = int(cy - h / 2)
    sx = int(cx + w / 2)
    sy = int(cy + h / 2)
    return px, py, sx, sy
def points_borders(ptos1x,ptos1y,ptos2x,ptos2y):
    ptos1x = np.array(ptos1x)
    p1x_l = np.amin(ptos1x)
    ptos1y = np.array(ptos1y)
    p1y_l = np.amin(ptos1y)
    ptos2x = np.array(ptos2x)
    p2x_l = np.amax(ptos2x)
    ptos2y = np.array(ptos2y)
    p2y_l = np.amax(ptos2y)

    return p1x_l,p1y_l,p2x_l,p2y_l
def crop_limits(p1x_l, p1y_l, p2x_l, p2y_l,x_aumt,y_aumt):
    limx1 = p1x_l - x_aumt
    limx2 = p2x_l + x_aumt
    limy1 = p1y_l - y_aumt
    limy2 = p2y_l + y_aumt
    # print(f'lim x1:{limx1},lim x2:{limx2},lim y1:{limy1}:lim y2 {limy2}')
    # redifiniendo limites:
    x_ch = 0
    y_ch = 0
    if limx1 < 0:
        limx1 = 0
        x_ch = 1
    if limx2 > ancho:
        limx2 = ancho
    if limy1 < 0:
        limy1 = 0
        y_ch = 1
    if limy2 > alto:
        limy2 = alto
    return limx1,limx2,limy1,limy2,x_ch,y_ch


def all_objects(img_d,labels,name):
    ptos1x = []
    ptos1y = []
    ptos2x = []
    ptos2y = []
    border_all = {}
    img = img_d
    if labels.shape[0] >= 2:
        final =np.zeros((labels.shape[0],5))
        for boxes in range(labels.shape[0]):
            ancho = img.shape[1]
            alto = img.shape[0]
            cx,cy,w,h,objeto =center_dimensions(labels[boxes],ancho,alto)
            p1x,p1y,p2x,p2y = borders(cx,cy,w,h)
            ptos1x.append(p1x)
            ptos1y.append(p1y)
            ptos2x.append(p2x)
            ptos2y.append(p2y)
            border_all[boxes] = [cx,cy,w,h,objeto]
        p1x_l, p1y_l, p2x_l, p2y_l = points_borders(ptos1x,ptos1y,ptos2x,ptos2y)
        #definicion de limites recorte
        x_aumt = int(0.02*ancho)
        y_aumt = int(0.02*alto)
        limx1,limx2,limy1,limy2,x_ch,y_ch = crop_limits(p1x_l, p1y_l, p2x_l, p2y_l,x_aumt,y_aumt)
        ancho_new = limx2 - limx1
        alto_new = limy2 - limy1
        img_resize = img[limy1:limy2,limx1:limx2]
        cv2.imwrite(path_save_images+name+'_all_b'+'.jpg', img_resize)
        #print(border_all)
        for key in border_all.keys():
            cx = border_all[key][0]
            cy = border_all[key][1]
            w_o = border_all[key][2]
            h_o = border_all[key][3]
            if x_ch ==0:
                cx = cx - limx1
            if y_ch ==0:
                cy = cy - limy1
            #cambio formato
            cx = cx/ancho_new
            cy = cy/alto_new
            w = w_o/ancho_new
            h = h_o/alto_new
            final[key] = [objeto,cx,cy,w,h]
        doc = open(path_save_labels+name+'_all_b'+'.txt', 'w')
        for n in range(final.shape[0]):
            objetost=int(final[n][0])
            cx =final[n][1]
            cy =final[n][2]
            w =final[n][3]
            h =final[n][4]
            final_alt = '{} {} {} {} {}\n'.format(objetost, cx,cy,w,h)
            doc.write(final_alt)
