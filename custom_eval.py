"""
Author: Zylo117
COCO-Style Evaluations

put images here datasets/your_project_name/annotations/val_set_name/*.jpg
put annotations here datasets/your_project_name/annotations/instances_{val_set_name}.json
put weights here /path/to/your/weights/*.pth
change compound_coef

"""

import json
import os
import numpy as np
import psutil

import argparse
import torch
import yaml
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, boolean_string


#python file parameters
#------------------------------------------------------------------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')
ap.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
ap.add_argument('-w', '--weights', type=str, default=None, help='/path/to/weights')
ap.add_argument('--nms_threshold', type=float, default=0.5, help='nms threshold, don\'t change it if not for testing purposes')
ap.add_argument('--cuda', type=boolean_string, default=True)
ap.add_argument('--device', type=int, default=0)
ap.add_argument('--float16', type=boolean_string, default=False)
ap.add_argument('--override', type=boolean_string, default=True, help='override previous bbox results file if exists')
ap.add_argument('--confidence_threshold', type=float, required=True, help='confidence threshold used in the prediction of bounding boxes')
ap.add_argument('--max_detections', type=int, required=True, help='maximum detections per image')
args = ap.parse_args()

compound_coef = args.compound_coef
nms_threshold = args.nms_threshold
use_cuda = args.cuda
gpu = args.device
use_float16 = args.float16
override_prev_results = args.override
project_name = args.project
weights_path = f'weights/efficientdet-d{compound_coef}.pth' if args.weights is None else args.weights
confidence_threshold = args.confidence_threshold
max_detections = args.max_detections
#------------------------------------------------------------------------------------------------------------------------------


#load default values and parameters
#------------------------------------------------------------------------------------------------------------------------------
print(f'running coco-style evaluation on project {project_name}, weights {weights_path}...')
params = yaml.safe_load(open(f'projects/{project_name}.yml'))
obj_list = params['obj_list']
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
#------------------------------------------------------------------------------------------------------------------------------



#Run the prediction of bounding boxes and store the results into a file
def evaluate_coco(img_path, set_name, image_ids, coco, model, conf_threshold):
    results = []

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    for image_id in tqdm(image_ids):
        image_info = coco.loadImgs(image_id)[0]
        image_path = img_path + image_info['file_name']

        ori_imgs, framed_imgs, framed_metas = preprocess(image_path, max_size=input_sizes[compound_coef])
        x = torch.from_numpy(framed_imgs[0])

        if use_cuda:
            x = x.cuda(gpu)
            if use_float16:
                x = x.half()
            else:
                x = x.float()
        else:
            x = x.float()

        x = x.unsqueeze(0).permute(0, 3, 1, 2)
        features, regression, classification, anchors = model(x)

        preds = postprocess(x,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            conf_threshold, nms_threshold)
        
        if not preds:
            continue

        preds = invert_affine(framed_metas, preds)[0]

        scores = preds['scores']
        class_ids = preds['class_ids']
        rois = preds['rois']

        if rois.shape[0] > 0:
            # x1,y1,x2,y2 -> x1,y1,w,h
            rois[:, 2] -= rois[:, 0]
            rois[:, 3] -= rois[:, 1]

            bbox_score = scores

            for roi_id in range(rois.shape[0]):
                score = float(bbox_score[roi_id])
                label = int(class_ids[roi_id])
                box = rois[roi_id, :]

                image_result = {
                    'image_id': image_id,
                    'category_id': label + 1,
                    'score': float(score),
                    'bbox': box.tolist(),
                }

                results.append(image_result)

    if not len(results):
        print('the model does not provide any valid output, check model architecture and the data input')
        return 0
        

    # write output
    filepath = f'{set_name}_bbox_results.json'
    if os.path.exists(filepath):
        os.remove(filepath)
    json.dump(results, open(filepath, 'w'), indent=4)

    return len(results)

#Run the evaluation of the model
def _eval(coco_gt, image_ids, pred_json_path, max_detect_list):
    # load results in COCO evaluation tool
    coco_pred = coco_gt.loadRes(pred_json_path)

    # run COCO evaluation
    print('BBox')
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.maxDets = max_detect_list
    #coco_eval.params.useCats = 0
    coco_eval.params.imgIds = image_ids
    
    # Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    coco_eval.evaluate()
    
    # Accumulate per image evaluation results and store the result in self.eval
    coco_eval.accumulate()
    
    # Compute and display summary metrics for evaluation results.
    coco_eval.summarize()
    
    #get precision and recall values from
    #   >>iou wherer the threshold is greater than 0.5
    #       from the values [0.5, 0.55, 0.6, ..., 0.95] --> a total of 10 values.
    #   >>I get all recall thresholds from the 101 interpolation of precision.
    #   >>category is the one from pineapple, actually there is only one category
    #       but, this software detect categories 0 and 1... where 0 is trash.
    #   >>area is related to 'all' from the values: [all, small, medium, large]
    #   >>get the highest max detections... from the values [0, 10, 100] or [10, 100, 1000] or ...
    iou_val = 0
    category_val = 1
    area_val = 0
    maxDet_val = 2
    
    #from the 101 precision vector, get the lowest precision
    precision_temp = coco_eval.eval["precision"][iou_val, :, category_val, area_val, maxDet_val]
    precision_result = precision_temp[np.where(precision_temp > 0)][-1]
    
    #get recall
    recall_result = coco_eval.eval["recall"][iou_val, category_val, area_val, maxDet_val]
    
    #print results
    return precision_result, recall_result


#LIMIT THE NUMBER OF CPU TO PROCESS THE JOB
def throttle_cpu(cpu_list):
    p = psutil.Process()
    for i in p.threads():
        temp = psutil.Process(i.id)
        temp.cpu_affinity([i for i in cpu_list])

#main method to be called
if __name__ == '__main__':
    throttle_cpu([28,29,30,31,32,33,34,35,36,37,38,39])
    
    SET_NAME = params['val_set']
    VAL_GT = f'datasets/{params["project_name"]}/annotations/instances_{SET_NAME}.json'
    VAL_IMGS = f'datasets/{params["project_name"]}/{SET_NAME}/'
    MAX_IMAGES = 10000
    coco_gt = COCO(VAL_GT)
    image_ids = coco_gt.getImgIds()[:MAX_IMAGES]
    
    #insert file header of results
    import csv
    with open(f'results/{params["project_name"]}_results_d{compound_coef}.csv', "w") as myfile:
        my_writer = csv.writer(myfile, delimiter=',', quotechar='"')
        my_writer.writerow(["max_detections_per_image", "num_detections", "nms_threshold", "confidence_threshold", "precision", "recall", "f1_score"])
    
    #create list for the max detections
    max_detect_list = []
    current_max = max_detections
    for i in range(3):
        max_detect_list.append(int(current_max))
        current_max = current_max / 10
    list.reverse(max_detect_list)
    
    #create a list or just the single confidence threshold
    if confidence_threshold == -1:
        confidence_list = np.linspace(.05, 0.95, int(np.round((0.95 - .05) / .05)) + 1, endpoint=True)
    else:
        confidence_list = [confidence_threshold]
    
    #--------------------------------------------
    #iterate over all different confidence levels
    for conf_thres in confidence_list:
    
        #run detections
        if override_prev_results or not os.path.exists(f'{SET_NAME}_bbox_results.json'):
            model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                         ratios=eval(params['anchors_ratios']), scales=eval(params['anchors_scales']))
            model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
            model.requires_grad_(False)
            model.eval()
    
            if use_cuda:
                model.cuda(gpu)
    
                if use_float16:
                    model.half()
                    
            #run the prediction of the bounding boxes and store results into a file
            pineapples_detected = evaluate_coco(VAL_IMGS, SET_NAME, image_ids, coco_gt, model, conf_thres)  
    
        #evaluate model using the ground truth and the predicted bounding boxes
        if(pineapples_detected > 0):
            p,r = _eval(coco_gt, image_ids, f'{SET_NAME}_bbox_results.json', max_detect_list)
            f1_result = (2.0 * p * r)/ (p + r)
        else:
            p = 0
            r = 0
            f1_result = 0
            
        print()
        print("===============================================================")
        print("Precision:" + str(p))
        print("Recall:" + str(r))
        print("===============================================================")
        
        #store results
        with open(f'results/{params["project_name"]}_results_d{compound_coef}.csv', "a") as myfile:
            my_writer = csv.writer(myfile, delimiter=',', quotechar='"')
            my_writer.writerow([max_detections, pineapples_detected, nms_threshold, conf_thres, p, r, f1_result])
    #--------------------------------------------
    
    #get the exact numbers of pineapples from the ground truth.
    #read the json
    import json
    with open(VAL_GT, "r") as read_file:
        data = json.load(read_file)

    #write how many pineapples there are (ground truth bounding boxes) in the results file.
    with open(f'results/{params["project_name"]}_results_d{compound_coef}.csv', "a") as myfile:
        my_writer = csv.writer(myfile, delimiter=',', quotechar='"')
        my_writer.writerow([""])
        my_writer.writerow(["Number of pineapples:", len(data["annotations"])])
