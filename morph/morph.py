# how to run
# python3 morph.py <sr-img1> <sr-img2> <alpha> <dest-img> <morph-points>
# alpha is the amoutn of morphing 0 <= alpa <= 1
# it assumes there is a morph.txt file 

import numpy as np
import cv2 as cv
from scipy.spatial import Delaunay
import sys

list_of_points_im1 = []
list_of_points_im2 = []

with open(sys.argv[5], 'r') as f:
	points = [x.strip().split(" ") for x in f.readlines()]

for i in range(0, 15):
	list_of_points_im1.append((int(points[i][0]), int(points[i][1])))

for i in range(15, 30):
	list_of_points_im2.append((int(points[i][0]), int(points[i][1])))

alpha = float(sys.argv[3])
lis_im_m = []

img1 = cv.imread(sys.argv[1])
img2 = cv.imread(sys.argv[2])

y_dest = min(img1.shape[0], img2.shape[0])
x_dest = min(img1.shape[1], img2.shape[1])

resized_img1 = cv.resize(img1, (x_dest, y_dest)) 
resized_img2 = cv.resize(img2, (x_dest, y_dest))

display_img1 = resized_img1.copy()
display_img2 = resized_img2.copy()

img_mph = np.zeros(resized_img2.shape, dtype = resized_img2.dtype)

common_list = []
common_list.append((0, 0))
common_list.append((0, y_dest//2))
common_list.append((0, y_dest-1))
common_list.append((x_dest//2, 0))
common_list.append((x_dest//2, y_dest-1))
common_list.append((x_dest-1, 0))
common_list.append((x_dest-1, y_dest//2))
common_list.append((x_dest-1, y_dest-1))

list_of_points_im1.extend(common_list)
list_of_points_im2.extend(common_list)

lis_im = []
for i in range(len(list_of_points_im1)):
	lis_im.append(((list_of_points_im1[i][0]+list_of_points_im2[i][0])/2,(list_of_points_im1[i][1]+list_of_points_im2[i][1])/2))
	lis_im_m.append(((alpha*list_of_points_im1[i][0]+(1-alpha)*list_of_points_im2[i][0]),(alpha*list_of_points_im1[i][1]+(1-alpha)*list_of_points_im2[i][1])))

tri = Delaunay(lis_im)

aff_1 = []
aff_2 = []

for triplet in tri.simplices:
	temp1 = np.float32([list_of_points_im1[triplet[0]],list_of_points_im1[triplet[1]],list_of_points_im1[triplet[2]]])
	temp2 = np.float32([list_of_points_im2[triplet[0]],list_of_points_im2[triplet[1]],list_of_points_im2[triplet[2]]])
	temp_m = np.float32([lis_im_m[triplet[0]],lis_im_m[triplet[1]],lis_im_m[triplet[2]]])
	r1 = cv.boundingRect(temp1)
	r2 = cv.boundingRect(temp2)
	r = cv.boundingRect(temp_m)
	t1Rect = []
	t2Rect = []
	tRect = []
	for i in range(0, 3):
		tRect.append(((temp_m[i][0] - r[0]),(temp_m[i][1] - r[1])))
		t1Rect.append(((temp1[i][0] - r1[0]),(temp1[i][1] - r1[1])))
		t2Rect.append(((temp2[i][0] - r2[0]),(temp2[i][1] - r2[1])))
	mk = np.zeros((r[3], r[2], 3), dtype = np.float32)
	cv.fillConvexPoly(mk, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0);
	img1Rect = resized_img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
	img2Rect = resized_img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]
	leng = (r[2], r[3])
	trns = cv.getAffineTransform(np.float32(t1Rect),np.float32(tRect))
	d_t = cv.warpAffine(img1Rect,trns,(leng[0],leng[1]),None, flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101 )
	trns = cv.getAffineTransform(np.float32(t2Rect),np.float32(tRect))
	d_t1 = cv.warpAffine(img2Rect,trns,(leng[0],leng[1]),None, flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101 )
	img_rec = alpha*d_t + (1-alpha)*d_t1
	img_mph[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img_mph[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mk ) + img_rec * mk


cv.imwrite(sys.argv[4], img_mph)
# cv.imshow('det',img_mph)
# cv.waitKey(0)