# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 10:17:31 2018

@author: tomcr00se
"""
import cv2
import argparse
import csv
import numpy as np
from scipy.spatial import Delaunay

#from ast import literal_eval

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image" ,required=True, help="Path to the image")
ap.add_argument("-i2", "--image2" ,required=True, help="Path to the image2")
args = vars(ap.parse_args())
 
# close all open windows
myFile1 = open('img13.csv', 'r')  
reader1 = csv.reader(myFile1)

myFile2 = open('img23.csv', 'r')  
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
image = cv2.imread(args["image"])
image2 = cv2.imread(args["image2"])


size = image.shape
rect = (0, 0, size[1], size[0])

read11 = []
read22 = []

for i in read1:
    print (tuple(map(int, i[1:-1].split(','))))
    read11.append(tuple(map(int, i[1:-1].split(','))))
    
size2 = image2.shape
rect2 = (0, 0, size2[1], size2[0])

for i in read2:
    read22.append(tuple(map(int, i[1:-1].split(','))))


d1 = Delaunay(read11)
d2 = Delaunay(read22)

# Draw delaunay triangles
# draw_delaunay( image, d1, read11);
# Draw delaunay triangles
# draw_delaunay( image2, d2, read22 );

# print subdiv.getTriangleList()

# print subdiv2.getTriangleList()
# # raw_input()
# print len(subdiv.getTriangleList())
# print len(subdiv2.getTriangleList())
 
cv2.imshow("image",image)
cv2.imshow("image2",image2)

def bary(x1,y1,x2,y2,x3,y3,x,y):
    det = (y2-y3)*(x1-x3)+(x3-x2)*(y1-y3)
    v1 = float((y2-y3)*(x-x3)+(x3-x2)*(y-y3))/float(det)
    v2 = float((y3-y1)*(x-x3)+(x1-x3)*(y-y3))/float(det)
    v3 = 1-v1-v2
    return [v1,v2,v3]

def sign(x1,y1,x2,y2,x3,y3):
    return (x1-x3)*(y2-y3) - (x2-x3)*(y1-y3)

def inSide(x,y,x1,y1,x2,y2,x3,y3):
    b1 = (sign(x,y,x1,y1,x2,y2) < 0.0)
    b2 = (sign(x,y,x2,y2,x3,y3) < 0.0)
    b3 = (sign(x,y,x3,y3,x1,y1) < 0.0)
    # print b1,b2,b3
    # print ((b1 == b2) and (b2 == b3))
    return (b1 == b2) and (b2 == b3) 

def warp(img1,img2,alpha,subd1,subd2,read11,read22):
    #get intermediate warp
    
    triangle1 = subd1.simplices
    triangle2 = subd2.simplices

    triangle3 = []
    imageFinal = np.ndarray(shape = (img1.shape[0],img1.shape[1],img2.shape[2]),dtype = np.uint8 )
    # counter = 0
    # triangle3 = alpha*triangle1 + (1-alpha)*triangle2
    read111 = np.array(read11,dtype=float)
    read222 = np.array(read22,dtype=float)

    # print triangle3
    print (len(triangle1))
    print (len(triangle3))

    size = img1.shape
    for counter in range(len(triangle3)):
        # print counter
        i = [ alpha*read11[triangle1[counter][0]][0]+(1-alpha)*read22[triangle1[counter][0]][0]  ]
        i.append(alpha*read11[triangle1[counter][0]][1]+(1-alpha)*read22[triangle1[counter][0]][1] )
        i.append(alpha*read11[triangle1[counter][1]][0]+(1-alpha)*read22[triangle1[counter][1]][0] )
        i.append(alpha*read11[triangle1[counter][1]][1]+(1-alpha)*read22[triangle1[counter][1]][1] )
        i.append(alpha*read11[triangle1[counter][2]][0]+(1-alpha)*read22[triangle1[counter][2]][0] )
        i.append(alpha*read11[triangle1[counter][2]][1]+(1-alpha)*read22[triangle1[counter][2]][1] )

        rect = [min(i[0],i[2],i[4]),min(i[1],i[3],i[5]),max(i[0],i[2],i[4]),max(i[1],i[3],i[5])]
        for x in range(int(rect[0]),int(rect[2])+1):
            for y in range(int(rect[1]),int(rect[3])+1):
                # print inSide(x,y,i[0],i[1],i[2],i[3],i[4],i[5])
                if inSide(x,y,i[0],i[1],i[2],i[3],i[4],i[5]):
                        # print -1
                        bari =  bary(i[0],i[1],i[2],i[3],i[4],i[5],x,y)
                        p1x =  bari[0]*read111[triangle1[counter][0]][0]+bari[1]*read111[triangle1[counter][1]][0]+bari[2]*read111[triangle1[counter][2]][0]
                        p1y =  bari[0]*read111[triangle1[counter][0]][1]+bari[1]*read111[triangle1[counter][1]][1]+bari[2]*read111[triangle1[counter][2]][1]
          
                        p2x =  bari[0]*read222[triangle1[counter][0]][0]+bari[1]*read222[triangle1[counter][1]][0]+bari[2]*read222[triangle1[counter][2]][0]
                        p2y =  bari[0]*read222[triangle1[counter][0]][1]+bari[1]*read222[triangle1[counter][1]][1]+bari[2]*read222[triangle1[counter][2]][1]

                        p1x1 = int(p1x)
                        p2x1 = int(p2x)

                        p1y1 = int(p1y)
                        p2y1 = int(p2y)
          
                        # try:
                        imageFinal[y][x] = (alpha*img1[p1y1][p1x1] + (1-alpha)*img2[p2y1][p2x1]).astype(dtype = np.uint8)    
                        # except IndexError:
                            # pass

    cv2.imshow("final",imageFinal)
    cv2.waitKey(0)

warp(image,image2,0.3,d1,d2,read11,read22)

cv2.waitKey(0)
