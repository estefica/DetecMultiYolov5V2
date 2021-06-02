import numpy as np
import cv2  # computer vision
import os
import divide_images
import one_shot_label
import data_img_v1


# Funciones cambio label YOLO to RECTANGLE, (p1x,p1y) :(s1x,s1y)
def xywhtoxyxy(cx, cy, w, h):
    px = int(cx - w / 2)
    py = int(cy - h / 2)
    sx = int(cx + w / 2)
    sy = int(cy + h / 2)
    return px, py, sx, sy


# RECTACNGULO A YOLO #p1 y p4 (p1x,p1y) :(s1x,s1y) a cx, cy, w, h
def xyxytoxywh(px, py, sx, sy):
    w = int(sx - px)
    h = int(sy - py)
    cx = int((w / 2) + px)
    cy = int((h / 2) + py)
    return cx, cy, w, h


# LOS CUATRO PUNTOS
def puntos_cuadro(px, py, sx, sy, w, h):
    p1x = px
    p1y = py
    p2x = px + w
    p2y = py
    p3x = px
    p3y = py + h
    p4x = sx
    p4y = sy
    # print(f'puntos:{p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y}')
    return p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y


def almacenar_bordes(b_f1x, b_f1y, b_f2x, b_f2y, b_f3x, b_f3y, b_f4x, b_f4y):
    bb.append(b_f1x)
    bb.append(b_f1y)
    bb.append(b_f2x)
    bb.append(b_f2y)
    bb.append(b_f3x)
    bb.append(b_f3y)
    bb.append(b_f4x)
    bb.append(b_f4y)
    return bb


def crear_doc_texto(b_f1x, b_f1y, b_f4x, b_f4y, nombre, lleno1, lleno2, limx, limy, final):
    global w, h, wabs, habs, final_alt
    [cx, cy, w, h] = xyxytoxywh(b_f1x, b_f1y, b_f4x, b_f4y)
    cx = abs(round((cx - limx) / ancho_deseado, 8))
    cy = abs(round((cy - limy) / alto_deseado, 8))
    wabs = w
    habs = h
    w = round(w / ancho_deseado, 8)
    h = round(h / alto_deseado, 8)
    if 0 <= cx <= 1 and 0 <= cy <= 1:

        if wabs >= 0.4 * w_total[indicep] and habs >= 0.25 * h_total[indicep]:

            w = 0.95 * w
            h = 0.95 * h
            final_alt = '{} {} {} {} {}'.format(int(objeto), cx, cy, w, h)
            # print(nombre.format(lleno1, lleno2))

            doc = open(nombre.format(lleno1, lleno2), 'w')
            if texto_m > 0:
                # print('000000000 final')
                # print(f'final respaldo: {final_respaldo}')
                doc.write(final_alt)
                doc.write(final_respaldo)
            else:
                doc.write(final_alt)
            doc.close()
        else:
            final_alt = ''
            pass

    else:
        print(f'no esta normlalizado {fn}')
    return w, h, wabs, habs, final_alt


def img_txt(final):
    w, h, wabs, habs, final = crear_doc_texto(b_f1x, b_f1y, b_f4x, b_f4y, path_save_filest + 'obj[{}]_im{}.txt',
                                              indicep, n, limites_ancho[x], limites_alto[y], final)
    # cv2.rectangle(imgResize, (b_f1x, b_f1y), (b_f4x, b_f4y), (0, 255, 0), 10)
    if wabs >= 0.4 * w_total[indicep] and habs >= 0.25 * h_total[indicep]:
        cv2.imwrite(path_save_files + 'obj[{}]_im{}.jpg'.format(indicep, n), imgcrop)
    return

def puntos_bounding(c):  # Ingresa un punto y determina los otros restantess
    global b_f1x, b_f1y, b_f2x, b_f2y, b_f3x, b_f3y, b_f4x, b_f4y, paso
    if c != [[(0, 0), (0, 0), (0, 0), (0, 0)]]:  # solo SI existe un punto
        if c[0][0] == (rec_x, puntos_y[aux]) and paso == 0:

            b_f1x = rec_x
            b_f1y = puntos_y[aux]
            if rec_x + w_total[indicep] >= limites_ancho[x + 1]:
                b_f2x = limites_ancho[x + 1]
            else:
                b_f2x = rec_x + w_total[indicep]
            b_f2y = b_f1y
            b_f3x = b_f1x
            if puntos_y[aux] + h_total[indicep] >= limites_alto[y + 1]:
                b_f3y = limites_alto[y + 1]
            else:
                b_f3y = puntos_y[aux] + h_total[indicep]
            b_f4x = b_f2x
            b_f4y = b_f3y
            paso = 1
            almacenar_bordes(b_f1x, b_f1y, b_f2x, b_f2y, b_f3x, b_f3y, b_f4x, b_f4y)
            img_txt(final)
            objeto_region[keyn] = (
                limites_alto[y], limites_alto[y + 1], limites_ancho[x], limites_ancho[x + 1], h_total[indicep],
                w_total[indicep], indicep, bb, n)

        if c[0][1] == (rec_x, puntos_y[aux]) and paso == 0:

            if rec_x - w_total[indicep] < limites_ancho[x]:
                b_f1x = limites_ancho[x]
            else:
                b_f1x = rec_x - w_total[indicep]
            b_f1y = puntos_y[aux]
            b_f2x = rec_x
            b_f2y = puntos_y[aux]
            b_f3x = b_f1x
            if b_f1y + h_total[indicep] >= limites_alto[y + 1]:
                b_f3y = limites_alto[y + 1]
            else:
                b_f3y = b_f1y + h_total[indicep]
            b_f4x = b_f2x
            b_f4y = b_f3y
            paso = 1
            almacenar_bordes(b_f1x, b_f1y, b_f2x, b_f2y, b_f3x, b_f3y, b_f4x, b_f4y)

            img_txt(final)
            objeto_region[keyn] = (
                limites_alto[y], limites_alto[y + 1], limites_ancho[x], limites_ancho[x + 1], h_total[indicep],
                w_total[indicep], indicep, bb, n)

        if c[0][2] == (rec_x, puntos_y[aux]) and paso == 0:

            b_f1x = rec_x
            if puntos_y[aux] - h_total[indicep] <= limites_alto[y]:
                b_f1y = limites_alto[y]
            else:
                b_f1y = puntos_y[aux] - h_total[indicep]
            if rec_x + w_total[indicep] >= limites_ancho[x + 1]:
                b_f2x = limites_ancho[x + 1]
            else:
                b_f2x = rec_x + w_total[indicep]
            b_f2y = b_f1y
            b_f3x = b_f1x
            b_f3y = puntos_y[aux]
            b_f4x = b_f2x
            b_f4y = b_f3y
            paso = 1
            almacenar_bordes(b_f1x, b_f1y, b_f2x, b_f2y, b_f3x, b_f3y, b_f4x, b_f4y)
            img_txt(final)
            objeto_region[keyn] = (
                limites_alto[y], limites_alto[y + 1], limites_ancho[x], limites_ancho[x + 1], h_total[indicep],
                w_total[indicep], indicep, bb, n)

        if c[0][3] == (rec_x, puntos_y[aux]) and paso == 0:

            # print('entre 3')
            if rec_x - w_total[indicep] < limites_ancho[x]:
                b_f1x = limites_ancho[x]
            else:
                b_f1x = rec_x - w_total[indicep]
            if puntos_y[aux] - h_total[indicep] < limites_alto[y]:
                b_f1y = limites_alto[y]
            else:
                b_f1y = puntos_y[aux] - h_total[indicep]
            b_f2x = rec_x
            b_f2y = b_f1y
            b_f3x = b_f1x
            b_f3y = puntos_y[aux]
            b_f4x = rec_x
            b_f4y = puntos_y[aux]
            paso = 1
            almacenar_bordes(b_f1x, b_f1y, b_f2x, b_f2y, b_f3x, b_f3y, b_f4x, b_f4y)
            img_txt(final)
            objeto_region[keyn] = (
                limites_alto[y], limites_alto[y + 1], limites_ancho[x], limites_ancho[x + 1], h_total[indicep],
                w_total[indicep], indicep, bb, n)

    return bb, c, final_alt

def limites_ancho_alto(alto, alto_deseado, ancho, ancho_deseado):
    for img_generada_a in range(int(alto / alto_deseado) + 1):
        img_generada_a *= alto_deseado
        limites_alto.append(img_generada_a)
        # print(limites_alto)
    for img_generada_an in range(int(ancho / ancho_deseado) + 1):
        img_generada_an *= ancho_deseado
        limites_ancho.append(img_generada_an)
        # print(limites_ancho)
    return limites_alto, limites_ancho

    ############################################### inicio ###########################################

def labels_multi():
    path_imagenes = '/content/prueba_data/'

    global path_save_files, w, h, path_save_filest, grid, w_total, h_total, indicep, final_respaldo, final, keyn
    global limites_alto, limites_ancho, puntos, puntos_x, puntos_y, w_total, h_total, num_objeto, n, c, cont, paso, objeto_region
    global rec_x, aux, x, y, bb, ancho_deseado, alto_deseado, objeto, texto_m, imgcrop,data_setbase
    alto_deseado = 720
    ancho_deseado = 960

    
    path_ima = '/content/prueba_data/'
    path_ima_t = '/content/prueba_data/'
    path_s_files = '/content/train/images/'
    path_s_filest = '/content/train/labels/'

    path_save_f = f'/content/random/images/'
    path_save_ft = f'/content/random/labels/'
    data_setbase = []
    dst_dir = ['/content/train/', '/content/valid/', '/content/test/','/content/random/']
    files_dst = ['images/', 'labels/']
    id_img = 0

    for i in dst_dir:
        try:
            os.mkdir(i)
        except Exception as e:
            pass
            # print(e)

    for i in dst_dir:
        for j in files_dst:
            try:
                os.mkdir(i + j)
            except Exception as e:
                pass

    # print('DEBERIA HACER EL ESCALADO A 1')

    lista_total_data = os.listdir(path_imagenes)
    for f in os.listdir(path_imagenes):
        
        if f.endswith('.jpg'):
            fn, ftext = os.path.splitext(f)
            
            if os.path.exists(path_imagenes + fn + '.txt'):
                id_img +=1
                img = cv2.imread(path_imagenes + f'{fn}.jpg')
                data_setbase.append(f)
                f = open(path_imagenes + fn + '.txt')
                labels = np.array([x.split() for x in f.read().strip().splitlines()],
                                  dtype=np.float32)  # labels
                if labels.shape[1] == 5:

                    #Escalado inicial
                    divide_images.escalado_inicial(fn, path_imagenes, path_save_f, path_save_ft,img,ancho_deseado,alto_deseado)
                    #solo el objeto
                    prox=1
                    one_shot_label.one_shot_imagen(prox,img,fn,labels)
                    #el objeto con bordes
                    prox = 1.5
                    one_shot_label.one_shot_imagen(prox, img,fn,labels)
                    prox = 2
                    one_shot_label.one_shot_imagen(prox, img,fn,labels)
                    prox = 2.5
                    one_shot_label.one_shot_imagen(prox, img,fn,labels)
                    data_img_v1.all_objects(img,labels,fn,id_img)
                    gridl = [g for g in range(2, int(img.shape[1] / ancho_deseado) + 1)]
                    for grid in gridl:
                        path_save_files =  path_save_f + f'{fn}_{grid}'
                        path_save_filest = path_save_ft + f'{fn}_{grid}'
                        limites_alto = []
                        limites_ancho = []
                        puntos = []
                        puntos_x = []
                        puntos_y = []
                        w_total = []
                        h_total = []
                        num_objeto = []
                        n = 0
                        cont = 0
                        paso = 0
                        c = [[(0, 0), (0, 0), (0, 0), (0, 0)]]
                        objeto_region = {}
                        resolucion_original_ancho = ancho_deseado * grid  # no mayor a 10
                        resolucion_original_alto = int(
                            resolucion_original_ancho * ((img.shape[0]) / (img.shape[1])))  # no mayor a 10
                        resolucion_original_alto = int(alto_deseado * (int(round(resolucion_original_alto / alto_deseado))))
                        imgResize = cv2.resize(img, (
                            resolucion_original_ancho, resolucion_original_alto))  # width and height
                        alto = int(imgResize.shape[0])
                        ancho = int(imgResize.shape[1])
                        ##################INICIO DEL PROGRAMA#############################################################
                        # Division de limites de nuevos recortes
                        limites_ancho_alto(alto, alto_deseado, ancho, ancho_deseado)
                        ######### BOUNDING BOXES POINTS######################
                        for boxes in range(labels.shape[0]):
                            linen = 0
                            ## NORMALIZADO A COORDENADAS REALES
                            cx_real = int(labels[boxes][1] * ancho)
                            cy_real = int(labels[boxes][2] * alto)
                            w_real = int(labels[boxes][3] * ancho)
                            h_real = int(labels[boxes][4] * alto)
                            objeto = labels[boxes][0]
                            px, py, sx, sy = xywhtoxyxy(cx_real, cy_real, w_real, h_real)
                            p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y = puntos_cuadro(px, py, sx, sy, w_real, h_real)
                            # p1:left/top,p2:right/top,p3:left/bottom,p4:right/bottom
                            puntos.append([(p1x, p1y), (p2x, p2y), (p3x, p3y), (p4x, p4y)])
                            puntos_x.append(p1x)
                            puntos_x.append(p2x)
                            puntos_x.append(p3x)
                            puntos_x.append(p4x)
                            puntos_y.append(p1y)
                            puntos_y.append(p2y)
                            puntos_y.append(p3y)
                            puntos_y.append(p4y)
                            w_total.append(w_real)
                            h_total.append(h_real)
                            comprobar = 0
                            if 0 > labels[boxes][1] > 1 or 0 > labels[boxes][2] > 1 or 0 > labels[boxes][3] > 1 or 0 > labels[boxes][4] > 1:
                                comprobar = 1  # SE CAMBIA SI NO ESTA NORMALIZADO

                        if comprobar == 0:
                            for x in range(len(limites_ancho) - 1):
                                # y sigue desde la parte superior de la imagen hacia abajoZ
                                multiples = 0  # variable por si hay más de un elemento en una imagen
                                final_respaldo = ''
                                final = ''
                                for y in range(len(limites_alto) - 1):
                                    c = [[(0, 0), (0, 0), (0, 0),
                                          (0, 0)]]  # CONTIENEN LOS PUNTOS OBTENIDOS DESDE EL .TXT

                                    bb = []  # GUARDA LOS NUEVOS PUNTOS DE CADA SEGMENTO DE IMAGEN
                                    n += 1  # ME INDICA EL NUMERO DE IMAGEN PERTENECIENTE A CADA NUEVO CUADRO
                                    cont = 0
                                    paso = 0  # HACE QUE NO SE VUELVA A REPETIR LA ASIGNACION DE CADA PUNTO EN LA IMAGEN
                                    texto_m = 0
                                    keyn = str(n)
                                    for rec_x in puntos_x:  # primero los puntos en X
                                        cont += 1
                                        aux = cont - 1
                                        if rec_x in range(limites_ancho[x], limites_ancho[
                                                                                x + 1] + 1):  # Punto entre el EJE X, DE EXTREMO A EXTREMO
                                            if puntos_y[aux] in range(limites_alto[y],
                                                                      limites_alto[y + 1] + 1):  # REVISO EJE Y
                                                multiples += 1

                                                for index1, value1 in enumerate(puntos):
                                                    for index2, value2 in enumerate(value1):
                                                        if value2 == (rec_x, puntos_y[aux]):
                                                            indice = index2  # IDENTIFICADOR DE LA COORDENADA A LA QUE PERTENECE P1, P2,P3,P4#IDENTIFICADOR DE CUAL OBJETO ES
                                                            indicep = index1
                                                imgcrop = imgResize[limites_alto[y]:limites_alto[y + 1],
                                                          limites_ancho[x]: limites_ancho[x + 1]]
                                                c[0][indice] = (rec_x, puntos_y[aux])
                                                # print(c)

                                                puntos_bounding(
                                                    c)  # OBTENGO IMAGENES Y LABELS DE LOS PUNTOS EXTREMOS P1,P2,P3,P4 y objeto region guarda a cada imagen

                                                if multiples >= 4:  # EN CASO DE QUE EXISTA MÁS DE UN OBJETO EN LA MISMA IMAGEN
                                                    # print('entre')
                                                    # print(n)
                                                    c = [[(0, 0), (0, 0), (0, 0), (0, 0)]]
                                                    multiples = 0
                                                    paso = 0
                                                    texto_m = 1
                                                    keyn = str(n) + str(multiples)
                                                    final_respaldo += '\n' + final_alt
                                                    bb = []

                                for key in objeto_region:
                                    num_objeto.append(objeto_region[key][6])
                                num_objeto = list(set(num_objeto))

                                for nm in num_objeto:
                                    hcmi = []
                                    limites_alto_complemento = []
                                    limites_ancho_complemento = []
                                    cont_puntos = 0
                                    reset = 0
                                    for key in objeto_region:
                                        if nm == objeto_region[key][6]:

                                            cont_puntos += 1
                                            if reset == 1:
                                                cont_puntos += 1
                                            if cont_puntos == 1:
                                                c1p1x = objeto_region[key][7][0]
                                                c1p1y = objeto_region[key][7][1]
                                                c1p2x = objeto_region[key][7][2]
                                                c1p2y = objeto_region[key][7][3]
                                                c1p3x = objeto_region[key][7][4]
                                                c1p3y = objeto_region[key][7][5]
                                                # print(f'cuadro 1 :( {c1p1x},{c1p1y}),({c1p2x},{c1p2y}),({c1p3x},{c1p3y})')
                                                borde_iz1 = objeto_region[key][2]
                                                borde_iz2 = objeto_region[key][3]
                                                borde_sup1 = objeto_region[key][0]
                                                borde_sup2 = objeto_region[key][1]
                                                hcmiaux1 = c1p3y - c1p1y
                                                anchoaux1 = c1p2x - c1p1x
                                                if anchoaux1 == objeto_region[key][5]:
                                                    # print('un solo ancho')
                                                    reset = 1
                                                if hcmiaux1 == objeto_region[key][4]:
                                                    pass
                                                    # print('un solo alto')

                                            if cont_puntos == 2:
                                                c3p1x = objeto_region[key][7][0]
                                                c3p1y = objeto_region[key][7][1]
                                                c3p2x = objeto_region[key][7][2]
                                                c3p2y = objeto_region[key][7][3]
                                                c3p3x = objeto_region[key][7][4]
                                                c3p3y = objeto_region[key][7][5]
                                                borde_inf1 = objeto_region[key][0]
                                                borde_inf2 = objeto_region[key][1]
                                                hcmiaux2 = c3p3y - c3p1y
                                                hparcial = hcmiaux1 + hcmiaux2
                                                cuadros_altof = int((objeto_region[key][4] - hparcial) / alto_deseado)
                                                # print(f'/////cuadro alto :{cuadros_altof}')

                                            if cont_puntos == 3:
                                                c2p1x = objeto_region[key][7][0]
                                                c2p1y = objeto_region[key][7][1]
                                                c2p2x = objeto_region[key][7][2]
                                                c2p2y = objeto_region[key][7][3]
                                                c2p3x = objeto_region[key][7][4]
                                                c2p3y = objeto_region[key][7][5]
                                                borde_der1 = objeto_region[key][2]
                                                borde_der2 = objeto_region[key][3]
                                                anchoaux2 = c2p2x - c2p1x
                                                cuadros_anchof = int(
                                                    (objeto_region[key][5] - (anchoaux1 + anchoaux2)) / ancho_deseado)
                                                imagen_numero = objeto_region[key][8]
                                                if reset == 1:
                                                    c3p1y = c2p1y
                                                    hcmiaux2 = c2p3y - c2p1y
                                                    hparcial = hcmiaux1 + hcmiaux2
                                                    cuadros_altof = int(
                                                        (objeto_region[key][4] - hparcial) / alto_deseado)
                                                    # print(f'/////cuadro alto :{cuadros_altof}')

                                                # print(f'numero de cuadros:{cuadros_altof,cuadros_anchof}')
                                                if cuadros_altof > 0:
                                                    # print(f'soy complemento:{imagen_numero}')
                                                    superiory = c1p3y
                                                    inferiory = c3p1y
                                                    lateralx = c1p2x
                                                    lateral2x = c2p1x
                                                    limites_alto_complemento.append(superiory)
                                                    limites_ancho_complemento.append(lateralx)
                                                    # izquierdo
                                                    for piezasy in range(cuadros_altof):
                                                        superiory += alto_deseado
                                                        limites_alto_complemento.append(superiory)
                                                    imagen_numero = 1

                                                    # cont_puntos
                                                    if reset == 0:
                                                        for iz in range(len(limites_alto_complemento) - 1):
                                                            imgcrop = imgResize[
                                                                      limites_alto_complemento[iz]:
                                                                      limites_alto_complemento[
                                                                          iz + 1],
                                                                      borde_iz1: borde_iz2]
                                                            crear_doc_texto(c1p1x, limites_alto_complemento[iz],
                                                                            borde_iz2,
                                                                            limites_alto_complemento[iz + 1],
                                                                            path_save_filest + 'obj{}_iz{}.txt',
                                                                            [objeto_region[key][6]], imagen_numero,
                                                                            borde_iz1,
                                                                            limites_alto_complemento[iz], final)

                                                            if wabs >= 0.4 * w_total[indicep] and habs >= 0.25 *h_total[indicep]:
                                                                cv2.imwrite(path_save_files + 'obj{}_iz{}.jpg'.format(
                                                                    [objeto_region[key][6]], imagen_numero), imgcrop)
                                                                imagen_numero += 1  # wabs >=0.38*w_total[indicep] and habs >=0.27*h_total[indicep]

                                                    # derecho
                                                    imagen_numero = 1
                                                    for der in range(len(limites_alto_complemento) - 1):
                                                        # cv2.rectangle(imgResize, (borde_der1, limites_alto_complemento[der]),
                                                        #            (c2p2x, limites_alto_complemento[der + 1]), (0, 255, 0),
                                                        #                15)
                                                        imgcrop = imgResize[
                                                                  limites_alto_complemento[der]:
                                                                  limites_alto_complemento[
                                                                      der + 1], borde_der1: borde_der2]
                                                        if reset == 0:
                                                            crear_doc_texto(borde_der1, limites_alto_complemento[der],
                                                                            c2p2x,
                                                                            limites_alto_complemento[der + 1],
                                                                            path_save_filest + 'obj{}_der{}.txt',
                                                                            [objeto_region[key][6]], imagen_numero,
                                                                            borde_der1,
                                                                            limites_alto_complemento[der], final)
                                                        if reset == 1:
                                                            crear_doc_texto(c1p1x, limites_alto_complemento[der], c2p2x,
                                                                            limites_alto_complemento[der + 1],
                                                                            path_save_filest + 'obj{}_der{}.txt',
                                                                            [objeto_region[key][6]], imagen_numero,
                                                                            borde_der1,
                                                                            limites_alto_complemento[der], final)

                                                        if wabs >= 0.4 * w_total[indicep] and habs >= 0.25 * h_total[indicep]:
                                                            cv2.imwrite(
                                                                path_save_files + 'obj{}_der{}.jpg'.format(
                                                                    [objeto_region[key][6]], imagen_numero),
                                                                imgcrop)
                                                            imagen_numero += 1

                                                if cuadros_anchof > 0:

                                                    ############## HORIZONTALES #######################################################
                                                    for piezasx in range(cuadros_anchof):
                                                        lateralx += ancho_deseado
                                                        limites_ancho_complemento.append(lateralx)
                                                    for sup in range(len(limites_ancho_complemento) - 1):
                                                        # cv2.rectangle(imgResize, (limites_ancho_complemento[sup], c1p1y),
                                                        #              (limites_ancho_complemento[sup + 1], borde_sup2),
                                                        #             (0, 255, 0), 15)
                                                        imgcrop = imgResize[borde_sup1: borde_sup2,
                                                                  limites_ancho_complemento[sup]:
                                                                  limites_ancho_complemento[
                                                                      sup + 1]]
                                                        crear_doc_texto(limites_ancho_complemento[sup], c1p1y,
                                                                        limites_ancho_complemento[sup + 1], borde_sup2,
                                                                        path_save_filest + 'obj{}_sup{}.txt',
                                                                        [objeto_region[key][6]],
                                                                        imagen_numero, limites_ancho_complemento[sup],
                                                                        borde_sup1, final)
                                                        if wabs >= 0.4 * w_total[indicep] and habs >= 0.25 * h_total[
                                                            indicep]:
                                                            cv2.imwrite(path_save_files + 'obj{}_sup{}.jpg'.format(
                                                                [objeto_region[key][6]], imagen_numero), imgcrop)
                                                            imagen_numero += 1

                                                    # inferior
                                                    imagen_numero = 1
                                                    for inf in range(len(limites_ancho_complemento) - 1):
                                                        # cv2.rectangle(imgResize, (limites_ancho_complemento[sup], borde_inf1),
                                                        #              (limites_ancho_complemento[sup + 1], c3p3y), (0, 255, 0),
                                                        #             15)
                                                        imgcrop = imgResize[borde_inf1: borde_inf2,
                                                                  limites_ancho_complemento[inf]:
                                                                  limites_ancho_complemento[
                                                                      inf + 1]]

                                                        crear_doc_texto(limites_ancho_complemento[sup], borde_inf1,
                                                                        limites_ancho_complemento[sup + 1], c3p3y,
                                                                        path_save_filest + 'obj{}_inf{}.txt',
                                                                        [objeto_region[key][6]],
                                                                        imagen_numero, limites_ancho_complemento[inf],
                                                                        borde_inf1, final)
                                                        if wabs >= 0.4 * w_total[indicep] and habs >= 0.25 * h_total[indicep]:
                                                            cv2.imwrite(path_save_files + 'obj{}_inf{}.jpg'.format(
                                                                [objeto_region[key][6]], imagen_numero), imgcrop)
                                                            imagen_numero += 1

                                                imagen_numero = 1
                                                if cuadros_anchof * cuadros_altof != 0:
                                                    for dx in range(len(limites_ancho_complemento) - 1):
                                                        for dy in range(len(limites_alto_complemento) - 1):
                                                            # cv2.rectangle(imgResize, (
                                                            # limites_ancho_complemento[dx], limites_alto_complemento[dy]), (
                                                            #              limites_ancho_complemento[dx + 1],
                                                            #             limites_alto_complemento[dy + 1]), (0, 255, 0), 15)
                                                            imgcrop = imgResize[
                                                                      limites_alto_complemento[dy]:
                                                                      limites_alto_complemento[
                                                                          dy + 1],
                                                                      limites_ancho_complemento[dx]:
                                                                      limites_ancho_complemento[
                                                                          dx + 1]]

                                                            crear_doc_texto(limites_ancho_complemento[dx],
                                                                            limites_alto_complemento[dy],
                                                                            limites_ancho_complemento[dx + 1],
                                                                            limites_alto_complemento[dy + 1],
                                                                            path_save_filest + 'obj{}_centro{}.txt',
                                                                            [objeto_region[key][6]],
                                                                            imagen_numero,
                                                                            limites_ancho_complemento[dx],
                                                                            limites_alto_complemento[dy], final)
                                                            if wabs >= 0.4 * w_total[indicep] and habs >= 0.25 * h_total[indicep]:
                                                                cv2.imwrite(
                                                                    path_save_files + 'obj{}_centro{}.jpg'.format(
                                                                        [objeto_region[key][6]], imagen_numero),
                                                                    imgcrop)
                                                                imagen_numero += 1

                else:
                    print('NUMERO DE PARAMETROS INCORRECTO')
            else:
                print(f'no existe archivo{fn}.txt ')
    divide_images.divide_img(data_setbase)
