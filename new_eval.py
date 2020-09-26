# -*- coding: utf-8 -*-

"""### Prepare the required repositories 
* https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch

* https://github.com/rafaelpadilla/Object-Detection-Metrics

# Executing the Object Detection Metrics, through Rafael's Padilla code

### The next cell will be the only one to be changed
"""

import os
import sys
import shutil
import cv2
import torch
from torch.backends import cudnn
import argparse

from backbone import EfficientDetBackbone
import matplotlib.pyplot as plt

import numpy as np
import yaml
import json
import pandas as pd
import csv 

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, boolean_string
###These lines need to be changed by the user################

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--project', type=str, required=True, default=None, help='project file that contains parameters')
ap.add_argument('-w', '--weights', type=str, required=True, default=None, help='name of the weights file') ## This file shuld be in EfficientDet/logs directory
ap.add_argument('--nms_threshold', type=float, default=0.4, help='nms threshold, don\'t change it if not for testing purposes')
ap.add_argument('-c', '--compound_coef', type=int, required=True, default=0, help='coefficients of efficientdet')
ap.add_argument('--confidence_threshold', type=float, default=0.0, help='confidence threshold used in the prediction of bounding boxes when is a single test')
ap.add_argument('--single_test', type=boolean_string, required=True, default=False,
                        help='true to set up a single evaluation with a specific confidence threshold or false for several tests in a range of 0.05 to 0.95 confidence threshold')
args = ap.parse_args()

project = args.project #This is the yml project name, also is the class name of the bboxes
wieghtsFileName = args.weights #The name of the weights file
nms_threshold = args.nms_threshold
coefficient = args.compound_coef ## EfficientDet coefficient
confidence_threshold = args.confidence_threshold
###########################################################


rootDir = os.getcwd()+'/' ##The directory where is going to be located the EfficientDet repo and Object Detection Metrics repo
#This directory is where the results of the evaluation will be stored
dirForGroundTruthAndDetections = rootDir+'eval'

#Download Object Detection Metrics
#if "Object-Detection-Metrics" not in rootDir:
#  !git clone --depth 1 https://github.com/rafaelpadilla/Object-Detection-Metrics
#else:
#  !git pull

"""## Building the groundtruth and detection files"""




def getImageDetections(imagePath,weights,nms_threshold,confidenceParam,coefficient):
  compound_coef = coefficient
  force_input_size = None  # set None to use default size
  img_path  = imagePath

  threshold = confidenceParam
  iou_threshold = nms_threshold

  use_cuda = True
  use_float16 = False
  cudnn.fastest = True
  cudnn.benchmark = True

  obj_list = ['pineapple']

  # tf bilinear interpolation is different from any other's, just make do
  input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
  input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size
  ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)

  if use_cuda:
      x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
  else:
      x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

  x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

  model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),

                              # replace this part with your project's anchor config
                              ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
                              scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

  model.load_state_dict(torch.load(rootDir+'logs/apple/'+weights))
  model.requires_grad_(False)
  model.eval()

  if use_cuda:
      model = model.cuda()
  if use_float16:
      model = model.half()

  with torch.no_grad():
      features, regression, classification, anchors = model(x)

      regressBoxes = BBoxTransform()
      clipBoxes = ClipBoxes()

      out = postprocess(x,
                        anchors, regression, classification,
                        regressBoxes, clipBoxes,
                        threshold, iou_threshold)

  out = invert_affine(framed_metas, out)

  from PIL import Image
  import PIL
      
  for i in range(len(ori_imgs)):
      if len(out[i]['rois']) == 0:
          continue
      detectionsList = []
      for j in range(len(out[i]['rois'])):
          (x1, y1, x2, y2) = out[i]['rois'][j].astype(np.int)
          detectionsList.append((float(out[i]['scores'][j]),x1, y1, x2, y2))
      return detectionsList

os.system('mkdir ' + dirForGroundTruthAndDetections)
os.system('mkdir ' + dirForGroundTruthAndDetections + '/' + project)
os.system('mkdir ' + dirForGroundTruthAndDetections + '/' + project  +'/groundtruths')
os.system('mkdir ' + dirForGroundTruthAndDetections + '/' + project  +'/detections')
os.system('mkdir ' + dirForGroundTruthAndDetections + '/' + project  +'/results')

#! mkdir '{dirForGroundTruthAndDetections}'
#! mkdir '{dirForGroundTruthAndDetections}/{project}'
#! mkdir '{dirForGroundTruthAndDetections}/{project}/groundtruths'
#! mkdir '{dirForGroundTruthAndDetections}/{project}/detections'
#! mkdir '{dirForGroundTruthAndDetections}/{project}/results'

def generateFiles(confidenceParam):
  params = yaml.safe_load(open(f'{rootDir}projects/{project}'+'.yml'))
  evaluationfolder = params['val_set']
  ##### Let´s create the Ground Truth files to evaluate them with the Object Detection Metrics
  dirGroundTruthPath = dirForGroundTruthAndDetections+'/'+project+'/groundtruths/'
  dirDetectionsPath = dirForGroundTruthAndDetections+'/'+project+'/detections/'
  with open(rootDir+'datasets/apple/annotations/instances_'+str(evaluationfolder)+'.json') as f:
    data = json.load(f)

  annotationsDataframe = pd.DataFrame.from_records(data['annotations'])
  imagesDataframe = pd.DataFrame.from_records(data['images'])

  qtyGroundTruthDets = 0
  qtyPredictedDets = 0
  for index,image in imagesDataframe.iterrows():
    imageName = image['file_name'][0:len(image['file_name'])-4]

    fileGroundTruth = open(dirGroundTruthPath+imageName+".txt","w")
    fileDetections = open(dirDetectionsPath+imageName+".txt","w")

    imageId = image['id']
    for ind,row in annotationsDataframe.loc[annotationsDataframe['image_id'] == imageId].iterrows():
      left = int(row['bbox'][0]) 
      top = int(row['bbox'][1])
      w = int(row['bbox'][2])
      h = int(row['bbox'][3])
      right = left + w
      bottom = top + h
      #<class_name> <left> <top> <right> <bottom>

      fileGroundTruth.write(project+" "+str(left)+" "+str(top)+" "+str(right)+" "+str(bottom)+"\n")
      qtyGroundTruthDets = qtyGroundTruthDets + 1
    
    fileGroundTruth.close()

    ##### Let´s create the detections files to evaluate them with the Object Detection Metrics
    
    defaultDatasetPath = rootDir+'datasets/'+project+'/'+evaluationfolder+'/'
    image_detections = getImageDetections(defaultDatasetPath+image['file_name'],wieghtsFileName,nms_threshold,confidenceParam,coefficient)
    for confidenceScore,xd1,yd1,xd2,yd2 in image_detections:
      fileDetections.write(project+" "+str(confidenceScore)+" "+str(xd1)+" "+str(yd1)+" "+str(xd2)+" "+str(yd2)+"\n")
      qtyPredictedDets = qtyPredictedDets + 1
    fileDetections.close()
  return (qtyGroundTruthDets,qtyPredictedDets)

"""## Methods to run the evaluation over the model"""

def generateResults():
  os.system('python '+rootDir+'Object-Detection-Metrics/pascalvoc.py -gt '+dirForGroundTruthAndDetections+'/'+project+'/groundtruths/ -det '+dirForGroundTruthAndDetections+'/'+project+'/detections/ -t '+str(nms_threshold)+' -gtformat xyrb -detformat xyrb -sp '+dirForGroundTruthAndDetections+'/'+project+'/results')

def collectRsultsFromFiles(): 
  with open(dirForGroundTruthAndDetections+'/'+project+'/results/results.txt',"r") as file1:
          FileasList = file1.readlines()

  precision = FileasList[8].split()[len(FileasList[8].split())-1]
  recall = FileasList[9].split()[len(FileasList[9].split())-1]
  precision = float(precision[1:len(precision)-2])
  recall = float(recall[1:len(recall)-2])
  f1Score = 2*((precision*recall)/(precision+recall))
  
  return (FileasList[6].split()[1],FileasList[7].split()[1],precision,recall,f1Score)


if __name__ == '__main__':
  if args.single_test == True:
    if args.confidence_threshold == 0.0:
      print("For a single test you should specify the parameter confidence_threshold")
      sys.exit()

    groundTruth, detected = generateFiles(args.confidence_threshold)
    generateResults()

    dataset,ap,precision,recall,f1Score = collectRsultsFromFiles()
    print('##############################################')
    print('Result over the dataset: '+ dataset)
    print('Quantity of ground truth boundig boxes: '+str(groundTruth))
    print('Quantity of predicted boundig boxes: '+str(detected))
    print('The average precision is (AP): '+ap)
    print('Precision: '+str(precision))
    print('Recall: '+str(recall))
    print('F1 Score: '+str(f1Score))
    #prcurveImg = cv2.imread(dirForGroundTruthAndDetections+'/'+project+'/results/'+project+'.png')
    #plt.imshow(prcurveImg)
    #plt.show()
  else:
    confidence_threshold = 0.05
    hop = 0.05
    with open(f'{dirForGroundTruthAndDetections}/{project}/{project}_results_d{coefficient}.csv', "w") as myfile:
          my_writer = csv.writer(myfile, delimiter=',', quotechar='"')
          my_writer.writerow(["num_detections", "nms_threshold", "confidence_threshold", "average_precision", "precision", "recall", "f1_score"])

    while confidence_threshold < 1:
      os.system('rm -r '+dirForGroundTruthAndDetections+'/'+project+'/results')
      os.system('mkdir '+dirForGroundTruthAndDetections+'/'+project+'/results')

      groundTruth, detected = generateFiles(confidence_threshold)
      generateResults()
      dataset,ap,precision,recall,f1Score = collectRsultsFromFiles()
      
      ##Aqui hay que guar los valores en el CSV
      with open(f'{dirForGroundTruthAndDetections}/{project}/{project}_results_d{coefficient}.csv', "a") as myfile:
                my_writer = csv.writer(myfile, delimiter=',', quotechar='"')
                my_writer.writerow([detected, nms_threshold, confidence_threshold,ap, precision,recall,f1Score])

      confidence_threshold = round(confidence_threshold + hop,2)