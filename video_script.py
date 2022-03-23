from ngransac_demo import matcher, matcher_video
import sys
#project_name = 'craft'
#sys.path.append(project_name)
#sys.path.append('craft/basenet')
import time
import matplotlib
import matplotlib.pylab as plt
plt.rcParams["axes.grid"] = False
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft.craft_utils
import craft.imgproc
# import craft.file_utils
import json
import zipfile
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


from craft.craft import CRAFT
from collections import OrderedDict

import random

def generate_random_colour():
  r = random.randint(0,255)
  g = random.randint(0,255)
  b = random.randint(0,255)
  rgb = (r,g,b)
  return rgb

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict
    
def find_box(point_coord, poly_dict):
  for box_id in poly_dict.keys():
    box = poly_dict[box_id]
    point = Point(point_coord)
    polygon = Polygon([tuple(box[0]), tuple(box[1]), tuple(box[2]), tuple(box[3])])
    if polygon.contains(point):
      return box_id
    else:
      continue
  return -1

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = craft.imgproc.resize_aspect_ratio(image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = craft.imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft.craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft.craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft.craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = craft.imgproc.cvt2HeatmapImg(render_img)

    # if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text

def predict_video(img1_pth = './images/2_a.jpg', img2_pth = './images/2_b.jpg', pretrained_model = './models/craft_mlt_25k.pth'):
  net = CRAFT()     # initialize

  # print('Loading weights from checkpoint (' + a + ')')
  # if args.cuda:
  net.load_state_dict(copyStateDict(torch.load(pretrained_model)))
  # else:
  #     net.load_state_dict(copyStateDict(torch.load(model, map_location='cpu')))

  # if args.cuda:
  net = net.cuda()
  net = torch.nn.DataParallel(net)
  cudnn.benchmark = False

  net.eval()


  good_matches = []
  for i in range(len(all_matches)):
    if inlier_mask[i] == 1:
      good_matches.append(all_matches[i])

  point_corresp = []
  for match in good_matches:
    p1 = kp1[match.queryIdx].pt
    p2 = kp2[match.trainIdx].pt
    point_corresp.append((p1,p2))
    
  # load data
  cap1 = cv2.VideoCapture(vid1_pth)
  cap2 = cv2.VideoCapture(vid2_pth)

  number_frame = 30.0 #higher frames better quality of the video
  video_size = (1280,720)
  fourcc = cv2.VideoWriter_fourcc(*'DIVX')
  out = cv2.VideoWriter('/content/result.mp4',fourcc, number_frame,video_size)
    
  while True:
        ret1,frame1 = cap1.read()
        ret2,frame2 = cap2.read() 

        print(ret1)
        print(ret2)
        
        if ret1 and ret2:
            
            image1 = cv2.resize(frame1, (640, 720))
            image2 = cv2.resize(frame2, (640, 720))

            print('here')

            # cv2_imshow(frame1)
            # cv2_imshow(frame2)
            inlier_mask, all_matches, kp1, kp2 = matcher_video(image1, image2, 'matching_op.png', 900, 900, '', 1000, './models/weights_e2e_E_r1.00_.net', False)

            bboxes1, polys1, score_text1 = test_net(net, image1, 0.4, 0.4, 0.4, True, False, None)
            bboxes2, polys2, score_text2 = test_net(net, image2, 0.4, 0.4, 0.4, True, False, None)
            
            polys1_dict = {}
            count1 = 0
            for matches1 in polys1:
              polys1_dict['a' + str(count1)] = matches1.astype('int16')
              count1 +=1

            polys2_dict = {}
            count2 = 0
            for matches2 in polys2:
              polys2_dict['b' + str(count2)] = matches2.astype('int16')
              count2 +=1

            relational_matrix = pd.DataFrame(np.zeros((len(polys2_dict), len(polys1_dict))), columns = polys1_dict.keys(), index= polys2_dict.keys())



            for corresp in point_corresp:
              box_id1 = find_box(corresp[0], polys1_dict)
              box_id2 = find_box(corresp[1], polys2_dict)
              if box_id1 == -1 or box_id2 == -1:
                continue
              else:
                relational_matrix[box_id1][box_id2] +=1


            final_corresp_boxes = []
            for column in relational_matrix:
              max_idx = relational_matrix[column].idxmax()
              max_val = relational_matrix[column].max()
              if max_val > 0:
                corr = (column, max_idx)
                final_corresp_boxes.append(corr)

            print(final_corresp_boxes[:5])


            from google.colab.patches import cv2_imshow
            for c in final_corresp_boxes:
              first, second = c
              bx1 = polys1_dict[first].reshape((-1, 1, 2))
              bx2 = polys2_dict[second].reshape((-1, 1, 2))
              color = generate_random_colour()
              image1 = cv2.polylines(image1, np.int32([bx1]), True, color, 4)
              image2 = cv2.polylines(image2, np.int32([bx2]), True, color, 4)
              #frame = np.concatenate((frame1, frame2), axis=1)
              frame = cv2.hconcat([frame1, frame2])
              #print(frame.shape)
              #break
              #cv2_imshow(frame)

              out.write(frame)

            # k = cv2.waitKey(5)

            # if k == ord("q"):
            #     break
            print('loop end')

        else:
            break
 
  
  
if __name__ == "__main__":
  predict_video()
