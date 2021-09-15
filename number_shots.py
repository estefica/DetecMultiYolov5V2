import imutils
import cv2
import numpy as np

def check_image(imagen):
  if imagen.shape[0]%32==0:
    new_w =imagen.shape[1]
    new_h =imagen.shape[0]
  if imagen.shape[1]%32!=0:
    new_w = int(round(imagen.shape[1]/512)*512)
  if imagen.shape[0]%32!=0:
    new_h = int(round(imagen.shape[0]/512)*512)
  return new_w,new_h

def check_imagef(imagen):
  if imagen.shape[1]%32!=0:
    new_wf = int(round(imagen.shape[1]/512)*512)
  if imagen.shape[0]%32!=0:
    new_hf = int(round(imagen.shape[0]/512)*512)
  if imagen.shape[0]%32==0:
    new_wf =imagen.shape[1]
    new_hf =imagen.shape[0]
  return new_wf,new_hf

def pyramid(image, scale = 1.5, minSize= (720,720)):
    yield image
    while True:
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        yield image

def sliding_window(image,stepSize, windowSize):
    for y in range(0,image.shape[0],stepSize):
        for x in range(0,image.shape[1],stepSize):
            yield(x,y,image[y:y + windowSize[1],x:x +windowSize[0]])

def img_num(path):

  image = cv2.imread(path) # <<<<<<---------------LEE LA IMAGEN
  (winW, winH) = (1024,1408)
  new_w= image.shape[1]
  new_h = image.shape[0]
  image1=image
  cont = 0
  win_min = 300
  images_new = {}
  escala = 0
  for i, resize in enumerate(pyramid(image, scale=1.6)):  # enumerate entrega indice y la imagen escalada
    escalay = (image1.shape[0]/resize.shape[0])*1 #conf scala en  y 
    escalax = 1*((image1.shape[1]/resize.shape[1])) #conf scala en  y 
    h_escala = resize.shape[0]
    w_escala = resize.shape[1]
    for (x, y, window) in sliding_window(resize, stepSize=350, windowSize=(winW, winH)):
      if window.shape[0] > win_min and window.shape[1] > win_min:
        cont += 1
        if window.shape[0] != winH or window.shape[1] != winW:
          pad_right = winW - window.shape[1]
          pad_bottom = winH - window.shape[0]
          window = np.pad(window, [(0, pad_bottom), (0, pad_right), (0, 0)])
        images_new[cont] = (window,x,y,escalay,escalax,h_escala,w_escala)
  print(cont)
  return cont, images_new

def vid_img_num(image1):
  (winW, winH) = (1024,1408)
  new_w= image1.shape[1]
  new_h = image1.shape[0]
  image = cv2.resize(image1, (new_w, new_h))
  cont = 0
  win_min = 300
  images_new = {}
  escala = 0
  for i, resize in enumerate(pyramid(image, scale=1.5)):  # enumerate entrega indice y la imagen escalada
    escalay = (image1.shape[0]/resize.shape[0])*1 #conf scala en  y 
    escalax = 1*((image1.shape[1]/resize.shape[1])) #conf scala en  y 
    h_escala = resize.shape[0]
    w_escala = resize.shape[1]
    for (x, y, window) in sliding_window(resize, stepSize=350, windowSize=(winW, winH)):
      if window.shape[0] > win_min and window.shape[1] > win_min:
        cont += 1
        if window.shape[0] != winH or window.shape[1] != winW:
          pad_right = winW - window.shape[1]
          pad_bottom = winH - window.shape[0]
          window = np.pad(window, [(0, pad_bottom), (0, pad_right), (0, 0)])
        images_new[cont] = (window,x,y,escalay,escalax,h_escala,w_escala)
  return cont, images_new
