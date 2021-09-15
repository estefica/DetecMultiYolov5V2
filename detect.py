import argparse
import time
from pathlib import Path
import scipy
import cv2
import torch
import math
import numpy as np
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression_inf, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import sys

def midist(punto1,punto2):
  distancia1 = math.sqrt((punto2[0]-punto1[0])**2 + (punto2[1]-punto1[1])**2)
  distancia2 = math.sqrt((punto2[2]-punto1[2])**2 + (punto2[3]-punto1[3])**2)
  distancia3 = math.sqrt((punto2[2]-punto1[2])**2 + (punto2[1]-punto1[1])**2)
  distancia4 = math.sqrt((punto2[0]-punto1[0])**2 + (punto2[3]-punto1[3])**2)
  return distancia1, distancia2,distancia3,distancia4 #s-izq,inf-der, sup-der,,inf -izq

def area_media(total):
  ancho = total[:,2:3]-total[:,0:1]
  alto = total[:,3:4]-total[:,1:2] 
  area = ancho*alto

  indice = np.where(np.all(area == [0], axis=1)) #encuentro el indice del que se parece
  area = np.delete(area, indice[0], 0)
  if 0 in area:
    area_cero = 0
  else:
    area_cero = 1
  media1 = (np.max(area))*0.35
  media2 = (np.mean(area))*0.5
  media = np.max([media1, media2])
  indices = []
  
  for a in range(area.shape[0]):
    if area[a]<media:
      indices.append(a)
  indices.sort(reverse=True)
  for a in indices:
    total = np.delete(total,a,0)
  return total

def sobrepos(punto1,punto2):
  
  distancia1 = abs(punto2[0]-punto1[0])
  distancia2 = abs(punto2[2]-punto1[2])
  box1_x1 = punto1[0]
  box1_x2 = punto1[2]
  box2_x1 = punto2[0]
  box2_x2 = punto2[2]
  box1_y1 = punto1[1]
  box1_y2 = punto1[3]
  box2_y1 = punto2[1]
  box2_y2 = punto2[3]
  x1 = max([box1_x1, box2_x1])
  y1 = max([box1_y1, box2_y1])
  x2 = min([box1_x2, box2_x2])
  y2 = min([box1_y2, box2_y2])

  intersection_side1 = (x2 - x1)
  intersection_side2 = (y2 - y1)
  if intersection_side1 < 0:
    intersection_side1 = 0
  if intersection_side2 < 0:
    intersection_side2 = 0
  intersection = intersection_side1 * intersection_side2
  box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
  box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

  distancia3 = intersection / (box1_area + box2_area - intersection + 1e-6)
  return distancia1,distancia2,distancia3

def overlap(resultados_pre):
  fin_np = []
  finales_t =[]
  b_fin = []
  resultados_pre=torch.stack(sorted(resultados_pre, key=lambda suma: suma[4], reverse = True))
  final_tensor = torch.empty((0,6),device='cuda:0')
  resultados_pre =resultados_pre.cpu().numpy()
  resultados_pre3 =resultados_pre.copy()
  resultados_pre4 =resultados_pre.copy()
  aux=1
  cont =0
  pares = True
  contador = 0
  contador1 =1
  comprobar =True
  while pares:
    if len(resultados_pre4)<=1 :
      if len(resultados_pre4) == 0:
        a = np.array([[0.0,0.0,0.0,0.0,0.0,0.0]])
        np.reshape(a, (6,))
        apilar = a
      else:
        apilar = resultados_pre3[contador]
      new_tensor = torch.from_numpy(apilar).to(device='cuda:0' )
      final_tensor = torch.vstack((final_tensor, new_tensor))
      break
    punto1 = (int(resultados_pre3[contador][0]), int(resultados_pre3[contador][1]),int(resultados_pre3[contador][2]),int(resultados_pre3[contador][3])) 
    
    umbral_cercania =abs(0.2*(int(resultados_pre3[contador][0])-int(resultados_pre3[contador][2])))
    umbral_overl = 0.1
    while comprobar:
      if len(resultados_pre4) == 1 :
        break
      punto2 = (int(resultados_pre3[contador1][0]), int(resultados_pre3[contador1][1]),int(resultados_pre3[contador1][2]),int(resultados_pre3[contador1][3]))
      distancia1,distancia2,distancia3 =  sobrepos(punto1, punto2)#izquierda,d2:derecha,overlap
      if distancia3>umbral_overl:
        if distancia1<umbral_cercania or distancia2<umbral_cercania:
          try:
            if aux ==1:
              fin_np = resultados_pre[contador]
              resultados_pre4 = np.delete(resultados_pre4,contador1, 0) #borro elemento en el indice y asigno nuevamente
              resultados_pre4 = np.delete(resultados_pre4,contador , 0)
              fin_np = np.vstack((fin_np, resultados_pre[contador1]))
              aux +=1
            else:
              indice = np.where(np.all(resultados_pre4 == resultados_pre3[contador1], axis=1)) #encuentro el indice del que se parece
              resultados_pre4 = np.delete(resultados_pre4, int(indice[0]), 0)
              fin_np = np.vstack((fin_np,resultados_pre3[contador1]))   
          except:
            pass
        else:
          indice = np.where(np.all(resultados_pre4 == resultados_pre3[contador1], axis=1))
          resultados_pre4 = np.delete(resultados_pre4,indice[0], 0)

      contador1 +=1
      if len(resultados_pre3) == contador1:
        break
    if not np.any(fin_np):
       new_tensor = torch.from_numpy(resultados_pre3[contador]).to(device='cuda:0' )
       final_tensor = torch.vstack((final_tensor, new_tensor))
       resultados_pre4 = np.delete(resultados_pre4,contador, 0)
       contador1=1
       resultados_pre3 = resultados_pre4
       resultados_pre = resultados_pre3
       aux = 1
       continue
    finales_t=np.unique(fin_np,axis=0)
    x1 = min(finales_t[:,0])
    y1 = min(finales_t[:,1])
    x2 = max(finales_t[:,2])
    y2 = max(finales_t[:,3])
    p = max(finales_t[:,4])
    cl = max(finales_t[:,5])
    box_aumento =np.array([x1,y1,x2,y2,p,cl])
    resultados_pre3 = np.vstack((box_aumento,resultados_pre4)) #(nuevotensor,losquenofueronborrados)
    fin_np =[]
    aux=1
    contador1=1
    resultados_pre = resultados_pre3
    resultados_pre4 = resultados_pre3
  return final_tensor

def tensores_juntos(suma,umbral_cercania):
    #print(f'\n umbral_Cercania:{umbral_cercania}')
    cont =0
    #suma es el tensor con las detecciones
    resultados_pre = torch.empty((0,6),device='cuda:0') #inicializoooo
    for x in range(suma.shape[0]-1):
        punto1 = (int(suma[x][0]),int(suma[x][1]),int(suma[x][2]),int(suma[x][3]))
        umbral_cercania =abs(0.2*(int(suma[x][1])-int(suma[x][3])))
        for y in range(suma.shape[0]-1-x):
            y_1 = y + x
            punto2 = (int(suma[y_1+1][0]), int(suma[y_1+1][1]),int(suma[y_1+1][2]),int(suma[y_1+1][3]))
            distancia1, distancia2,distancia3,distancia4 = midist(punto1,punto2)
            if distancia1<umbral_cercania or distancia2<umbral_cercania or distancia3<umbral_cercania or distancia4<umbral_cercania:
                cont +=1  
                if cont==1:
                    resultados_pre = suma[x]
                    resultados_pre = torch.vstack((resultados_pre,suma[y_1+1]))
                if cont !=1:
                    resultados_pre = torch.vstack((resultados_pre,suma[x]))
                    resultados_pre = torch.vstack((resultados_pre,suma[y_1+1]))

    return resultados_pre


def bounding_separados(resultados_pre,suma):
  ########### variables inicializadas
  aviso = 0
  finales_t = {}
  fin_np = []
  final_tensor = torch.empty((0,6),device='cuda:0')
  cont =0
  ###########
  for x in suma:
      for y in resultados_pre:
        if torch.equal(x,y):
          aviso += 1
          break
      if aviso==0:
        final_tensor= torch.vstack((final_tensor,x))
      aviso=0
    # final_tensor: contiene los elementos no repetidos
  resultados_pre = torch.unique(resultados_pre, dim=0)
  return resultados_pre,final_tensor

def boundig_finales(final_tensor,resultados_pre,umbral_cercania):
  fin_np = []
  finales_t =[]
  b_fin = []

  resultados_pre =resultados_pre.cpu().numpy()
  resultados_pre3 =resultados_pre.copy()
  resultados_pre4 =resultados_pre.copy()
  aux=1
  cont =0
  pares = True
  contador = 0
  contador1 =1
  comprobar =True
  while pares:
    if len(resultados_pre4)==1:
      new_tensor = torch.from_numpy(resultados_pre3[contador]).to(device='cuda:0' )
      final_tensor = torch.vstack((final_tensor, new_tensor))
      break
    punto1 = (int(resultados_pre3[contador][0]), int(resultados_pre3[contador][1]),int(resultados_pre3[contador][2]),int(resultados_pre3[contador][3])) 
    umbral_cercania =abs(0.2*(int(resultados_pre3[contador][1])-int(resultados_pre3[contador][3])))
    while comprobar:
      punto2 = (int(resultados_pre3[contador1][0]), int(resultados_pre3[contador1][1]),int(resultados_pre3[contador1][2]),int(resultados_pre3[contador1][3]))
      distancia1,distancia2,distancia3,distancia4 =  midist(punto1, punto2)#distancias diagonales de los puntos p1 y p2
      if distancia1<umbral_cercania  or distancia2<umbral_cercania or distancia3<umbral_cercania or distancia4<umbral_cercania:
    
        try:
          if aux ==1:
            fin_np = resultados_pre[contador]
            resultados_pre4 = np.delete(resultados_pre4,contador1, 0) #borro elemento en el indice y asigno nuevamente
            resultados_pre4 = np.delete(resultados_pre4,contador , 0)
            fin_np = np.vstack((fin_np, resultados_pre[contador1]))
            aux +=1
          else:
            indice = np.where(np.all(resultados_pre4 == resultados_pre3[contador1], axis=1)) #encuentro el indice del que se parece
            resultados_pre4 = np.delete(resultados_pre4, int(indice[0]), 0)
            fin_np = np.vstack((fin_np,resultados_pre3[contador1]))   
        except:
          pass
      contador1 +=1
      if len(resultados_pre3) == contador1:
        break
    if not np.any(fin_np):
       new_tensor = torch.from_numpy(resultados_pre3[contador]).to(device='cuda:0' )
       final_tensor = torch.vstack((final_tensor, new_tensor))
       resultados_pre4 = np.delete(resultados_pre4,contador, 0)
       contador1=1
       resultados_pre3 = resultados_pre4
       resultados_pre = resultados_pre3
       aux = 1
       continue

    finales_t=np.unique(fin_np,axis=0)
    x1 = min(finales_t[:,0])
    y1 = min(finales_t[:,1])
    x2 = max(finales_t[:,2])
    y2 = max(finales_t[:,3])
    p = max(finales_t[:,4])
    cl = max(finales_t[:,5])
    box_aumento =np.array([x1,y1,x2,y2,p,cl])
    resultados_pre3 = np.vstack((box_aumento,resultados_pre4)) #(nuevotensor,losquenofueronborrados)
    fin_np =[]
    aux=1
    contador1=1
    resultados_pre = resultados_pre3
    resultados_pre4 = resultados_pre3
  #print(final_tensor)
  return final_tensor

def detect():
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    #definicion de tensor que almacenara a las detecciones
    p_out = torch.zeros([1,6]).to(device)
    #contador de fragmentos
    split_count = 0
    #Save results
    tensor_test = []
    for path, img, im0s, vid_cap,flag_im,x_cor,y_cor,escalay,escalax,alton,anchon in dataset:
        #print(f'\n imagen : {im0s.shape}')
        split_count +=1
        #print(f'\n Imagen :{split_count}')
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        # Apply NMS
        pred,p_out = non_max_suppression_inf(alton,anchon,pred,p_out,flag_im,x_cor,y_cor,escalay,escalax,split_count,opt.conf_thres, opt.iou_thres, classes_inf=opt.classes_inf, agnostic_nms_inf=opt.agnostic_nms_inf)
        #<<<<<<<<<<<<<<<<<<<<<<<<<<--------------------- PREDICCIONES  FINAL
        if flag_im !=1:
            continue
        if flag_im ==1:
          flag_im =0
          split_count =0
          umbral_cercania = im0s.shape[0]*0.1 #este ya no uso!
          resultados_pre = tensores_juntos(pred[0],umbral_cercania)
          resultados_pre,final_tensor = bounding_separados(resultados_pre,pred[0])
          #print('!!!!!!!!!!!!!! soy resultados pred arriba')
          if resultados_pre.shape[0] != 0:
            final_tensor = boundig_finales(final_tensor,resultados_pre,umbral_cercania)
            pred[0]=final_tensor
          else:
            pass
          try:
            pred[0] = overlap(pred[0])
            pass
          except Exception as e:
              print(e)
          p_aux = pred[0]
          pre_aux =p_aux.cpu().numpy() #numpy -- le hago numpy
          try:
            indice = np.where(np.all(pre_aux == [0,0,0,0,0,0], axis=1)) #encuentro el indice del que se parece
            pre_aux = np.delete(pre_aux, indice[0], 0)
            pred[0] = pre_aux
          except Exception as e:
              print(e)
          pre_aux = torch.from_numpy(pre_aux).to(device) #torch tensor
          pred[0] = pre_aux
          pre_aux =p_aux.cpu().numpy()
          if pre_aux.shape[0]>1:#usa un numpy
            try:
              pre_aux = area_media(pre_aux)
            except Exception as e:
              print(e)
        try:
            indice = np.where(np.all(pre_aux == [0,0,0,0,0,0], axis=1)) #encuentro el indice del que se parece
            pre_aux = np.delete(pre_aux, indice[0], 0)
            pred[0] = pre_aux
        except Exception as e:
          print(e)
        pre_aux = torch.from_numpy(pre_aux).to(device)
        pred[0] = pre_aux#pred_prueba#
        tensor_test.append(pred)
        t2 = time_synchronized()
        # Apply Classifier
        # Apply Classifier
        if classify:
            #print('\n entre a clasiffy')
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                #det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or opt.save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = f'{names[c]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=3)
                        if opt.save_crop:
                            save_one_box(xyxy, im0s, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            p_out = torch.empty((0,6),device='cuda:0')
            pre_aux = np.array([0,0,0,0,0,0])

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes_inf', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms_inf', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
