import cv2
import argparse
import csv
import numpy as np
from scipy.spatial import Delaunay
from numpy.linalg import pinv,norm

#from ast import literal_eval

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image" ,required=True, help="Path to the image")
ap.add_argument("-i2", "--image2" ,required=True, help="Path to the image2")
ap.add_argument("-d", "--dest" ,required=True, help="Path to the Dest image")
args = vars(ap.parse_args())
 

#######################################
# refPt = []
 
def click_and_record(event, x, y, flags, param):
    # grab references to the global variables
    global refPt
 
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append((x, y))
        cv2.rectangle(image, (refPt[-1][0]-2,refPt[-1][1]-2), (refPt[-1][0]+2,refPt[-1][1]+2), (0, 255, 0), 2)
        cv2.imshow("image", image)

# # load the image, clone it, and setup the mouse callback function
# image = cv2.imread(args["image"],cv2.IMREAD_UNCHANGED)
# if(image.shape[2] == 4):
#     ind = image[:,:,3] == 0
#     image[ind] = [150,150,150,0]
# clone = image.copy()
# cv2.namedWindow("image")
# cv2.setMouseCallback("image", click_and_record)

# # keep looping until the 'q' key is pressed
# while True:
#     # display the image and wait for a keypress
#     cv2.imshow("image", image)
#     key = cv2.waitKey(1) & 0xFF
 
#     # if the 'r' key is pressed, reset the cropping region
#     if key == ord("r"):
#         image = clone.copy()
 
#     # if the 'c' key is pressed, break from the loop
#     elif key == ord("c"):
#         break

# # close all open windows

# # print(refPt)
# myFile = open('t1.csv', 'w')  
# wr = csv.writer(myFile)
# wr.writerows([refPt])
# myFile.flush()
# myFile.close()
# cv2.destroyAllWindows()

#######################################

# refPt = []
# image = cv2.imread(args["image2"],cv2.IMREAD_UNCHANGED)
# if(image.shape[2] == 4):
#     ind = image[:,:,3] == 0
#     image[ind] = [150,150,150,0]
# clone = image.copy()
# cv2.namedWindow("image")
# cv2.setMouseCallback("image", click_and_record)

# # keep looping until the 'q' key is pressed
# while True:
#     # display the image and wait for a keypress
#     cv2.imshow("image", image)
#     key = cv2.waitKey(1) & 0xFF
 
#     # if the 'r' key is pressed, reset the cropping region
#     if key == ord("r"):
#         image = clone.copy()
 
#     # if the 'c' key is pressed, break from the loop
#     elif key == ord("c"):
#         break

# # close all open windows

# print(refPt)
# myFile = open('t2.csv', 'w')  
# wr = csv.writer(myFile)
# wr.writerows([refPt])
# myFile.flush()
# myFile.close()
# cv2.destroyAllWindows()
#######################################
# close all open windows
myFile1 = open('t1.csv', 'r')  
reader1 = csv.reader(myFile1)

myFile2 = open('t2.csv', 'r')  
reader2 = csv.reader(myFile2)

read1 = []
for row in reader1:
    read1.append(row)
read2 = []
for row in reader2:
    read2.append(row)

read1 = read1[0]
read2 = read2[0]
#print read1
#print read2



# Check if a point is inside a rectangle
def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

# Draw delaunay triangles
def draw_delaunay(img, d, points ) :
    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in d.simplices :
        pt1 = (points[t[0]][0], points[t[0]][1])
        pt2 = (points[t[1]][0], points[t[1]][1])
        pt3 = (points[t[2]][0], points[t[2]][1])
         
        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)
 
# Define colors for drawing.
delaunay_color = (255,255,255)
points_color = (0, 0, 255)

# load the image, clone it, and setup the mouse callback function
image = cv2.imread(args["image"],cv2.IMREAD_UNCHANGED)
image2 = cv2.imread(args["image2"],cv2.IMREAD_UNCHANGED)


size = image.shape
rect = (0, 0, size[1], size[0])

read11 = []
read22 = []

for i in read1:
    # print (tuple(map(int, i[1:-1].split(','))))
    temp = tuple(map(int, i[1:-1].split(',')))
    read11.append((temp[1],temp[0]))


print(read11)
print(read1)
print("---")
size2 = image2.shape
rect2 = (0, 0, size2[1], size2[0])

for i in read2:
    temp = tuple(map(int, i[1:-1].split(',')))
    read22.append((temp[1],temp[0]))

# cv2.imshow("image",image)
# cv2.imshow("image2",image2)
ind = image2[:,:,3] == 0
# image2[ind] = 0

# print(image.shape)
# print(image2.shape)

mask = np.zeros([image.shape[0],image.shape[1],4])

final_points = np.zeros([2,2])
final_points[:,0] = np.array(read11[0])
final_points[:,1] = np.array(read11[1])
# final_points[:,2] = np.array(read11[2])
initial_points = np.ones([3,2])
initial_points[:2,0] = np.array(read22[0])
initial_points[:2,1] = np.array(read22[1])

final_points = final_points.astype("float64")
initial_points = initial_points.astype("float64")
tinitial_points = np.copy(initial_points)
initial_points = initial_points[:2,:]
scale = norm(final_points[:,0]-final_points[:,1])/norm(initial_points[:2,0]-initial_points[:2,1])
initial_points = initial_points*scale
# print(norm(final_points[:,0]-final_points[:,1]))
# print(norm(initial_points[:2,0]-initial_points[:2,1]))
pf = final_points[:,0] - final_points[:,1]
pi = initial_points[:,0] - initial_points[:,1]
# print(norm(pf),norm(pi))
# print(pf,pi)
thetas = (pinv(np.array([[pi[0], -pi[1]],[pi[1], pi[0]]]))).dot(pf)
rot = np.zeros([2,2])
rot[:,0] = thetas
rot[0,1] = -thetas[1]
rot[1,1] = thetas[0]
# print(rot)
# print(pf)
# print(rot.dot(pi))
c = final_points[:,0] - rot.dot(initial_points[:,0])
# print(initial_points[:,0])
# print()
tfmatrix = np.zeros([2,3])
tfmatrix[:,:2] = rot
tfmatrix[:,2] = c
x = np.ones([3])
x[0:2] = initial_points[:,0]
# print(tfmatrix.dot(x))
# print(final_points[:,0])
# print(tfmatrix)
# print(scale)
# tfmatrix = tfmatrix*scale
# print(tfmatrix)
# print("------------------")
# print(final_points[:,0])
# print(x)
# print(tinitial_points[:,0])
# print(tfmatrix.dot(tinitial_points[:,0]))
# print(c)
# print(rot.dot(pi))
# print(thetas)
# print(norm(thetas))
# exit()
# initial_points[:2,2] = np.array(read22[2])

# final_points = np.zeros([2,3])
# final_points[:,0] = np.array(read11[0])
# final_points[:,1] = np.array(read11[0]) + np.array([0,10])
# final_points[:,2] = np.array(read11[2])
# initial_points = np.ones([3,3])
# initial_points[:2,0] = np.array(read22[0])
# initial_points[:2,1] = np.array(read22[0])
# initial_points[:2,2] = np.array(read22[2])


# image2[158,230] = [255,0,0,1]
# image2[157,230] = [255,0,0,1]
# image2[158,231] = [255,0,0,1]
# image2[157,231] = [255,0,0,1]
# image2[243,378] = [255,0,0,1]
# image2[242,378] = [255,0,0,1]
# image2[243,377] = [255,0,0,1]
# image2[242,377] = [255,0,0,1]
# inv_init = inv(initial_points)
# print(final_points)
# print(initial_points)
# tfmatrix = final_points.dot(pinv(initial_points)) 
# print(tfmatrix)
# print(tfmatrix.dot(initial_points))

mrect = [0,0,mask.shape[0],mask.shape[1]]
# print(mrect)
for i in range(0,image2.shape[0]):
    for j in range(0,image2.shape[1]):
        if(image2[i][j][3] == 0):
            continue
        p = (np.round(tfmatrix.dot((np.array([i*scale,j*scale,1.0]))).astype("float64"))).astype('int')
        # print(p)
        if rect_contains(mrect,p):
            # print("here")

            mask[p[0],p[1]] = image2[i,j]
            # if(i == 243 and j == 378):
            #     print("-----------------")
            #     print(p)
            #     print(mask[p[0],p[1]])
            #     print(image2[i,j])
            #     print(p)
ind2 = mask[:,:,3] != 0
# print(np.sum(ind2))
# print(ind2.shape)
# print(mask.shape)
# print(image.shape)
# print(mask[157,228])
image[ind2] = mask[ind2,:3]
# cv2.imshow("mask",mask)
# cv2.imwrite("mask.png",mask)
cv2.imwrite(args["dest"],image)

# cv2.imshow("image",image)
# cv2.imshow("image2",image2)
# cv2.waitKey(0)
