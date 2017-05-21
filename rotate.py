import cv2
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from numpy import *
import math
from shapely.geometry import MultiPoint,mapping,Polygon
from pdb import set_trace

file_path = './icdar15/train_images'
save_rotate_path = './icdar15/train_images_rotate'
save_crop_path = './icdar15/train_images_rotate_crop'
vis_save_path = './icdar15/train_images_rotate_crop_vis_label'
gt_path = './icdar15/train_gts'
new_gt_path = './icdar15/new_train_gts'
def mkdir(path):
  if not os.path.exists(path):
    os.mkdir(path)
    
mkdir(save_rotate_path)
mkdir(save_crop_path)
mkdir(vis_save_path)
mkdir(new_gt_path)

def rotatedRectWithMaxArea(w, h, angle):
  """
  Given a rectangle of size wxh that has been rotated by 'angle' (in
  radians), computes the width and height of the largest possible
  axis-aligned rectangle (maximal area) within the rotated rectangle.
  """
  if w <= 0 or h <= 0:
    return 0,0

  width_is_longer = w >= h
  side_long, side_short = (w,h) if width_is_longer else (h,w)

  # since the solutions for angle, -angle and 180-angle are all the same,
  # if suffices to look at the first quadrant and the absolute values of sin,cos:
  sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
  if side_short <= 2.*sin_a*cos_a*side_long:
    # half constrained case: two crop corners touch the longer side,
    #   the other two corners are on the mid-line parallel to the longer line
    x = 0.5*side_short
    wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
  else:
    # fully constrained case: crop touches all 4 sides
    cos_2a = cos_a*cos_a - sin_a*sin_a
    wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

  return (wr,hr)


def rotate_point(origin, point, angle):
  """
  Rotate a point counterclockwise by a given angle around a given origin.

  The angle should be given in radians.
  """
  angle = math.radians(angle)
  ox, oy = origin
  px, py = point

  qx = ox + math.cos(-angle) * (px - ox) - math.sin(-angle) * (py - oy)
  qy = oy + math.sin(-angle) * (px - ox) + math.cos(-angle) * (py - oy)
  return [qx, qy]

def rotate_image(image, boxes, angle):
  """
  Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
  (in degrees). The returned image will be large enough to hold the entire
  new image, with a black background
  """  
  # Get the image size
  # No that's not an error - NumPy stores image matricies backwards
  image_size = (image.shape[1], image.shape[0])
  image_center = tuple(np.array(image_size) / 2)

  # Convert the OpenCV 3x2 rotation matrix to 3x3
  rot_mat = np.vstack(
      [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
  )
  #print 'rot_mat',rot_mat,'\n',rot_mat.shape

  rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])
  #print 'rot_mat_notranslate',rot_mat_notranslate,'\n',rot_mat_notranslate.shape
  # Shorthand for below calcs
  image_w2 = image_size[0] * 0.5
  image_h2 = image_size[1] * 0.5
  image_center = [image_w2,image_h2]
  # Obtain the rotated coordinates of the image corners
  rotated_coords = [
      (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
      (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
      (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
      (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
  ]
  # Find the size of the new image
  x_coords = [pt[0] for pt in rotated_coords]
  x_pos = [x for x in x_coords if x > 0]
  x_neg = [x for x in x_coords if x < 0]
  
  y_coords = [pt[1] for pt in rotated_coords]
  y_pos = [y for y in y_coords if y > 0]
  y_neg = [y for y in y_coords if y < 0]
  
  right_bound = max(x_pos)
  left_bound = min(x_neg)
  top_bound = max(y_pos)
  bot_bound = min(y_neg)
  
  new_w = int(abs(right_bound - left_bound))
  new_h = int(abs(top_bound - bot_bound))
  
  # We require a translation matrix to keep the image centred
  trans_mat = np.matrix([
      [1, 0, int(new_w * 0.5 - image_w2)],
      [0, 1, int(new_h * 0.5 - image_h2)],
      [0, 0, 1]
  ])
  #print 'trans_mat',trans_mat,'\n',trans_mat.shape,'\n'
  
  # Compute the tranform for the combined rotation and translation
  affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]
  #print 'affine_mat',affine_mat,'\n',affine_mat.shape,'\n'
  box_num = len(boxes)
  box_results = np.zeros((box_num,4,2),dtype=np.int32)
  for k,box in enumerate(boxes):
    #box = np.array(box).reshape(4, 2, 1)
    box = np.array(box).reshape(4, 2)
    #box_center = cv2.minAreaRect(box)[0]
    #box_repeat = np.repeat(box[:, :,np.newaxis], 3, axis=0)
    #print 'box',box,'\n',box.shape
    #box_result = np.matrix(affine_mat) * np.matrix(box)
    rotated_box =[
     rotate_point(image_center,box[0],angle),
     rotate_point(image_center,box[1],angle),
     rotate_point(image_center,box[2],angle),
        rotate_point(image_center,box[3],angle)]
    
    x1 = rotated_box[0][0]+ new_w * 0.5 -image_w2
    x2 = rotated_box[1][0]+ new_w * 0.5 -image_w2
    x3 = rotated_box[2][0]+ new_w * 0.5 -image_w2
    x4 = rotated_box[3][0]+ new_w * 0.5 -image_w2
    y1 = rotated_box[0][1]+ new_h * 0.5 -image_h2
    y2 = rotated_box[1][1]+ new_h * 0.5 -image_h2
    y3 = rotated_box[2][1]+ new_h * 0.5 -image_h2
    y4 = rotated_box[3][1]+ new_h * 0.5 -image_h2
    box_result = np.array([int(x1),int(y1),int(x2),int(y2),int(x3),int(y3),int(x4),int(y4)]).reshape(4,2)
    box_results[k] = box_result  
  # Apply the transform
  result = cv2.warpAffine(image,affine_mat,(new_w, new_h),flags=cv2.INTER_LINEAR)  
  return [result,box_results]

def perp( a ) :
  b = empty_like(a)
  b[0] = -a[1]
  b[1] = a[0]
  return b

# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
# return 
def seg_intersect(line1, line2) :
  da = line1[1] - line1[0]
  db = line2[1] - line2[0]
  dp = line1[0] - line2[0]
  dap = perp(da)
  denom = dot( dap, db)
  num = dot( dap, dp )
  intersect = (num / denom.astype(float))*db + line2[0]
  intersect = [int(a) for a in intersect]
  return intersect

def common_line(dot1,dot2,width,height):
  if dot1[0]==dot2[0]:
    if (dot1[0]- 0)<1e-6 or (dot1[0]- width)<1e-6:
      return True
  elif dot1[1]==dot2[1]:
    if (dot1[1]- 0)<1e-6 or (dot1[1]- height)<1e-6:
      return True
  else:
    return False

def line_length(dot1,dot2):
  return math.sqrt((dot1[0]-dot2[0])**2+ (dot1[1]-dot2[1])**2)

def clockwise(poly_bad, poly_ori):
  
  poly_good = np.zeros((4,2),dtype=np.int32)
  discard_indexs = []
  exist_indexs = []
  for i in range(4):
    for j in range(4):
      if list(poly_bad[i]) == list(poly_ori[j]):
        exist_index = j
        poly_good[exist_index] = poly_bad[i]
        exist_indexs.append(exist_index)
        discard_indexs.append(i)
      else:
        continue
  to_do_bad_index = sorted(list(set(range(4))^set(discard_indexs)))
  print 'to_do_bad_index',to_do_bad_index
  to_fill_good_index = sorted(list(set(range(4))^set(exist_indexs)))
  print 'to_fill_good_index',to_fill_good_index
  if len(to_fill_good_index) == 0:
    return poly_good
  elif len(to_fill_good_index) == 1:
    poly_good[to_fill_good_index[0]] = poly_bad[to_do_bad_index[0]]
    return poly_good
  else:
    print 'bad index len',len(to_do_bad_index),to_do_bad_index
    dot1 = poly_bad[to_do_bad_index[0]]
    dot2 = poly_bad[to_do_bad_index[1]]
    if dot1[0]<=dot2[0]:
      if dot1[1]<=dot2[1]:
        poly_good[to_fill_good_index[0]] = dot1
        poly_good[to_fill_good_index[1]] = dot2
      else:
        poly_good[to_fill_good_index[0]] = dot2
        poly_good[to_fill_good_index[1]] = dot1
    else:
      if dot1[1]<=dot2[1]:
        poly_good[to_fill_good_index[0]] = dot1
        poly_good[to_fill_good_index[1]] = dot2
      else:
        poly_good[to_fill_good_index[0]] = dot2
        poly_good[to_fill_good_index[1]] = dot1
    return poly_good
      
  
def intersect(poly1,poly2,width,height):
  polygon1 = Polygon(poly1).convex_hull
  polygon2 = Polygon(poly2).convex_hull
  intersect = polygon2.intersection(polygon1)
  #set_trace()
  dots = mapping(intersect)['coordinates'][0]
  #print dots
  poly = [] 
  flag=False
  valid = False
  index = 0
  if len(dots)<5:
    return valid,poly
  elif len(dots)==5:
    print 'len(dots)<5',len(dots)
    for i in range(len(dots)):   
      dot = [int(dots[i][0]),int(dots[i][1])]
      dot = [int(dots[i][0]),int(dots[i][1])]
      #print dot
      poly.append(dot)
    new_poly = poly[0:4]
  else:
    for i in range(len(dots)-1):
      #print 'dots[i],dots[i+1]',dots[i],dots[i+1]
      if common_line(dots[i],dots[i+1],width,height):
        #print 'common_line:True'
        flag = True  
        #print 'flag',flag 
        if i == 0:
          index = 0  
        elif i == 4:
          index = 4               
        elif line_length(dots[i],dots[i-1])>line_length(dots[i+1],dots[i-1]):          
          index = i+1         
        else:
          index = i              
      dot = [int(dots[i][0]),int(dots[i][1])]
      print 'dot',dot
      poly.append(dot)
    if flag:
      if index == 0:
        new_poly = poly[1:5]
      elif index == 1:
        new_poly = [poly[0]]+poly[2:5]
      elif index == 4:
        new_poly = poly[0:4]
      elif index == 2:
        new_poly = poly[0:2]+poly[3:5]
      elif index == 3:
        new_poly = poly[0:3]+poly[4:5]
      else:
        new_poly = poly[0:4]
    else:
      new_poly = poly[0:4]
  
  print 'valid',valid
  print 'new_poly',new_poly
  valid = True
  new_poly = np.array(new_poly,dtype=np.int32).reshape(4,2)
  #print 'new_poly',new_poly.shape
  new_poly = clockwise(new_poly,poly2)
  return valid,new_poly
  
def crop_around_center(image, boxes, width, height):
  """
  Given a NumPy / OpenCV 2 image, crops it to the given width and height,
  around it's centre point
  """
  
  image_size = (image.shape[1], image.shape[0])
  image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))
  
  if(width > image_size[0]):
      width = image_size[0]
  
  if(height > image_size[1]):
      height = image_size[1]
  
  xx1 = int(image_center[0] - width * 0.5)
  xx2 = int(image_center[0] + width * 0.5)
  yy1 = int(image_center[1] - height * 0.5)
  yy2 = int(image_center[1] + height * 0.5)
  
  #box_crop_results = np.zeros((boxes.shape[0],4,2),dtype=np.int32)
  box_crop_results = []
  '''
  left_line = np.array([[0,0],[0,height]],dtype=np.int32)
  right_line = np.array([[width,0],[width,height]],dtype=np.int32)
  top_line = np.array([[0,0],[width,0]],dtype=np.int32)
  bottom_line = np.array([[0,height],[width,height]],dtype=np.int32)
  '''
  img_poly = np.array([[0,0],[0,height],[width,0],[width,height]],dtype=np.int32)
  for k in range(boxes.shape[0]):
    x1 = boxes[k][0][0]-image_size[0]*0.5+width * 0.5
    x2 = boxes[k][1][0]-image_size[0]*0.5+width * 0.5
    x3 = boxes[k][2][0]-image_size[0]*0.5+width * 0.5
    x4 = boxes[k][3][0]-image_size[0]*0.5+width * 0.5
    y1 = boxes[k][0][1]-image_size[1]*0.5+height * 0.5
    y2 = boxes[k][1][1]-image_size[1]*0.5+height * 0.5
    y3 = boxes[k][2][1]-image_size[1]*0.5+height * 0.5
    y4 = boxes[k][3][1]-image_size[1]*0.5+height * 0.5
    box_result = np.array([int(x1),int(y1),int(x2),int(y2),int(x3),int(y3),int(x4),int(y4)]).reshape(4,2)
    flag = 0
    border_flag = 0
    for i in range(4):
      if box_result[i][0]<0:
        flag +=1
      if box_result[i][0]>width:
        flag += 1
      if box_result[i][1]<0:
        flag +=1
      if box_result[i][1]>height:
        flag +=1
      if box_result[i][0]==0 or box_result[i][0]==width or box_result[i][1]== 0 or box_result[i][1]== height:
        border_flag +=1
    flag +=border_flag        
    if flag>=3:
     continue
    elif flag == 0: 
      box_crop_results.append(box_result) 
    else:    
      valid,box_result = intersect(img_poly,box_result,width,height)
      if valid:
        box_crop_results.append(box_result) 
  return image[yy1:yy2, xx1:xx2],box_crop_results

def read():
  """
  Demos the largest_rotated_rect function
  """
  for i in range(1,1001):
    img_name = 'img_'+str(i)+'.jpg'
    gt_name = 'gt_img_'+str(i)+'.txt'
    print img_name
    img_path = os.path.join(file_path,img_name)
    with open(os.path.join(gt_path,gt_name),'r') as f:
      gt_lines = [o.decode('utf-8-sig').encode('utf-8') for o in f.readlines()]
    gt_strs = [g.strip().split(',')[-1] for g in gt_lines]
    gt_coors = [g.strip().split(',')[0:8] for g in gt_lines]
    #gt_coors = [int(g) for g in gt_coors]
    for ii,g in enumerate(gt_coors):
      gt_coors[ii] = [int(a) for a in g]
    #gt_coors = np.array(gt_coor,dtype=np.int32)
    #print 'gt_coors',gt_coors
    image = cv2.imread(img_path)    
    image_height, image_width = image.shape[0:2]  
    angles = [-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90]
    
    for j in angles:
      print 'angle',j
      image_orig = np.copy(image)
      [image_rotated,boxes_rotated] = rotate_image(image, gt_coors, j)
      image_rotated_cropped,boxes_rotated_cropped = crop_around_center(
          image_rotated,
          boxes_rotated,
          *rotatedRectWithMaxArea(
              image_width,
              image_height,
              math.radians(j)
          )
      )    
      new_img_name = 'img_'+str(i)+'_'+str(j)+'.jpg'
      
      for index, gt in enumerate(gt_coors):
        gt = np.array(gt).reshape(4, 2)
        
      for index in range(boxes_rotated.shape[0]):
        #print 'plot boxes_rotated ',boxes_rotated[index]
        gt_rotated = boxes_rotated[index].reshape(4, 2)
      result_lines = [] 
      for index in range(len(boxes_rotated_cropped)):
        #print 'plot boxes_rotated ',boxes_rotated[index]
        gt_rotated_cropped = boxes_rotated_cropped[index]
        print gt_rotated_cropped
        result_line = []
        for p in range(4):
          for q in range(2):
            result_line.append(gt_rotated_cropped[p][q])
        print 'result_line',result_line
        result_line = [str(a) for a in result_line]
        
        result_lines.append(','.join(result_line))
      with open(os.path.join(new_gt_path,'gt_img_'+str(i)+'_'+str(j)+'.txt'),'w') as f:
        f.write('\r\n'.join(result_lines))
      cv2.imwrite(os.path.join(save_rotate_path,new_img_name),image_rotated) 
      cv2.imwrite(os.path.join(save_crop_path,new_img_name), image_rotated_cropped) 
      
    print "Done"



if __name__ == "__main__":
    read()

  