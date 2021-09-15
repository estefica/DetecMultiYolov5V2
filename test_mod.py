import numpy as np
import torch
import os
from collections import Counter




# diccionario: { 'nombre': #detecciones}


#lectura de predicciones
# gt = [clase, cx,cy,w,h,conf]

#######################################################################################################################
#########################################################################################################################3

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct Labels of Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """

    # Slicing idx:idx+1 in order to keep tensor dimensionality
    # Doing ... in indexing if there would be additional dimensions
    # Like for Yolo algorithm which would have (N, S, S, 4) in shape
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # Need clamp(0) in case they do not intersect, then we want intersection to be 0
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)
#######################################################################################################################

def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.95, box_format="midpoint", num_classes=20
):
    """
    Calculates mean average precision 
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes
    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        
        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            print('no hay verdades')
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            #print(f'\n\n numero de imagenes: {num_gts}')
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                #print('\n')
                #print('\n####')
                #print(gt)
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        print(f'tp:{TP_cumsum}')

        FP_cumsum = torch.cumsum(FP, dim=0)
        print(f'fp:{FP_cumsum}')
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        print(f'recalls:{recalls}')
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        print(f'precision: {precisions}')
        precision_media = torch.mean(precisions)
        print(f'precision: {precision_media}<<<<<<---------MEAN--------------')
        recalls = torch.cat((torch.tensor([0]), recalls))
        print(f'recalls_2:{recalls}')
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))
        #print(average_precisions)


    return sum(average_precisions) / len(average_precisions)

#########################################################################################################################3


def read_labels(idx, detections,labels,tipo ='real'):
  for boxes in range(labels.shape[0]):
        clase = labels[boxes][0]
        cx = labels[boxes][1]
        cy = labels[boxes][2]
        w = labels[boxes][3]
        h = labels[boxes][4]
        if tipo =='detect':
          #print(labels)
          #print(boxes)
          prob = labels[boxes][5]
        else:
          prob = 1
        detect = [idx,clase,prob,cx,cy,w,h]
        #detect = torch.from_numpy(detect).to(device='cuda:0' )
        detections.append(detect)
        #print(detect)
        #print(detections)
  return detections
########################################################################################################3333



def bbxes(path_gt,path_pred):
  lista_gt = os.listdir(path_gt)
  lista_pr = os.listdir(path_pred)
  detections = []
  predictions = []
  idx = 0
  
  for f in os.listdir(path_gt):
      if f.endswith('.txt') and f in lista_pr:
        idx += 1
        fn, ftext = os.path.splitext(f)
        f = open(path_gt + fn + '.txt')
        #print(fn)
        labels_gt = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
        detections = read_labels(idx, detections,labels_gt, tipo = 'real') # GROUND
        #print('reales')
        f = open(path_pred + fn + '.txt')
        labels_pred = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
        #print(labels_pred)
        predictions = read_labels(idx, predictions,labels_pred, tipo = 'detect') #PREDICTIONS
        #print('detecciones')
  return detections,predictions

########################################################################


path_gt = '/content/zopo/' 
path_pred = '/content/DetecMultiYolov5V2/runs/detect/exp11/labels/'
print('\n YOLOv5m6 -0.5') 
detections, predictions = bbxes(path_gt,path_pred)
map_c = mean_average_precision(predictions,detections, iou_threshold=0.5, box_format="midpoint", num_classes=1)
print(f'map_c : {map_c}')
print('\n YOLOv5m6 -0.65')
path_pred = '/content/DetecMultiYolov5V2/runs/detect/exp12/labels/' 
detections, predictions = bbxes(path_gt,path_pred)
map_c = mean_average_precision(predictions,detections, iou_threshold=0.65, box_format="midpoint", num_classes=1)
print(f'map_c : {map_c}')

path_gt = '/content/veg/' 
print('\n YOLOv5m -0.5')
path_pred = '/content/DetecMultiYolov5V2/runs/detect/exp13/labels/' 
detections, predictions = bbxes(path_gt,path_pred)
map_c = mean_average_precision(predictions,detections, iou_threshold=0.5, box_format="midpoint", num_classes=1)
print(f'map_c : {map_c}')


print('\n YOLOv5m -0.65')
path_pred = '/content/DetecMultiYolov5V2/runs/detect/exp14/labels/' 
detections, predictions = bbxes(path_gt,path_pred)
map_c = mean_average_precision(predictions,detections, iou_threshold=0.65, box_format="midpoint", num_classes=1)
print(f'map_c : {map_c}')


path_gt = '/content/sol/' 

print('\n YOLOv5s -0.5')
path_pred = '/content/DetecMultiYolov5V2/runs/detect/exp15/labels/' 
detections, predictions = bbxes(path_gt,path_pred)
map_c = mean_average_precision(predictions,detections, iou_threshold=0.65, box_format="midpoint", num_classes=1)
print(f'map_c : {map_c}')


print('\n YOLOv5s -0.65')
path_pred = '/content/DetecMultiYolov5V2/runs/detect/exp16/labels/' 
detections, predictions = bbxes(path_gt,path_pred)
map_c = mean_average_precision(predictions,detections, iou_threshold=0.65, box_format="midpoint", num_classes=1)
print(f'map_c : {map_c}')

'''

print('\n YOLOv5s6 -0.5')
path_pred = '/content/DetecMultiYolov5V2/runs/detect/exp8/labels/' 
detections, predictions = bbxes(path_gt,path_pred)
map_c = mean_average_precision(predictions,detections, iou_threshold=0.5, box_format="midpoint", num_classes=1)
print(f'map_c : {map_c}')



print('\n YOLOv5s6 -0.65')
path_pred = '/content/DetecMultiYolov5V2/runs/detect/exp7/labels/' 
detections, predictions = bbxes(path_gt,path_pred)
map_c = mean_average_precision(predictions,detections, iou_threshold=0.65, box_format="midpoint", num_classes=1)
print(f'map_c : {map_c}')'''
