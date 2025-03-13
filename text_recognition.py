import numpy as np
from skimage import io
import os
from PIL import Image
from PIL import ImageOps
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.collections import PatchCollection

image_path = 'images/PandP_handwritten.jpeg'

def grayscale(a):
  '''
  a - rbg image as numpy array
  returns greyscale image
  '''
  p,q, _ = np.shape(a)
  m=np.zeros((p,q))
  for i in range(len(m)):
      for j in range(len(m[0])):
          m[i][j] = np.average(a[i][j])
  return m


def binary(a, T=200):# takes np array image
  '''
  a - numpy array of greyscale image
  T = threshold
  return binary image based on threshold
  '''
  m=a.copy()
  for i in range(len(m)):
      for j in range(len(m[0])):
          if m[i][j] >T:
            m[i][j]=255
          else:
            m[i][j]=0
  return m


def line_coords(coords):
    '''
    coords - cooridinate of edges in line
    returns line boundry rectangle
    '''
    ymin=coords[0][0][0]
    ymax=coords[-1][0][0]
    xmin=20000
    xmax=0
    for i in coords:
        for j in i:
            if j[1] >xmax:
                xmax=j[1]
            if j[1] < xmin:
                xmin=j[1]

    return [ymin,ymax+2,xmin+1,xmax]

def char_coords(coords, line):
    '''
    coords - cooridinate of edges in character
    returns character boundry rectangle
    '''
    xmin=coords[0][0][1]
    xmax=coords[-1][0][1]
    ymin=20000
    ymax=0
    for i in coords:
        for j in i:
            if j[0] >ymax:
                ymax=j[0]
            if j[0] < ymin:
                ymin=j[0]

    return [line[0]+ymin+1,line[0]+ymax, line[2]+xmin,line[2]+xmax+2]

def line_segment(a):
  '''
  a - nparray of binary image
  returns - boundaries of each line
  '''
  
  coords=[] #coordinates with text, on edge change
  xy_rect = [] # line detection info
  
  for i in range(len(a)): #for each row in image
    coo=[]
    flag=0
    for c in a[i]:
      if c<200: # check row has any text in it
        flag=1
        break
    
    if flag==1: # row had text somewhere
      for b in range(len(a[i])): # look for edge changes
        
        # no
        if a[i][b]>200: # if no text in current col
          try:
            if a[i][b+1] <200:  # if text in column to the right
              coo.append([i,b+1]) # add to list of edge changed
          except:
            pass

        if a[i][b]<200: #if text in current column
          try:
            if a[i][b+1] > 200: # if next column empty
              coo.append([i,b]) # add to list
    
          except:
            pass
    else: # row had no text

      if len(coords)>0: # end line
          xy_rect.append(line_coords(coords))
          coords=[]
    if len(coo)>0:
      coords.append(coo)
    
  return xy_rect

def char_segment(line, image):
  '''
  line - boundary of line
  image - nparray of binary image
  returns - boundaries of each character
  '''

  
  coords=[] #coordinates with text, on edge change
  xy_rect = [] # line detection info
  spaces = []
  a = image[line[0]:line[1],line[2]:line[3]+3]# added 3 so that last character was detected - todo find better fix
  space_count = 0
  spaces = []
  
  for i in range(len(a[0,:])): #for each col in image
    coo=[]
    flag=0
    for c in a[:,i]:
      if c<200: # check col has any text in it
        if space_count > 5:
          spaces.append(i+line[2]-space_count/2)
        space_count = 0
        flag=1
        break

    
    if flag==1: # col had text somewhere
      for b in range(len(a[:,i])): # look for edge changes
        
        # no
        if a[b][i]>200: # if no text in current row
          try:
            if a[b+1][i] <200:  # if text in row to the below
              coo.append([b+1, i]) # add to list of edge changed
          except:
            pass

        if a[b][i]<200: #if text in current row
          try:
            if a[b+1][i] > 200: # if next row empty
              coo.append([b,i]) # add to list
    
          except:
            pass
    else: # col had no text
      space_count +=1

      if len(coords)> 0: # end line
          xy_rect.append(char_coords(coords, line))
          coords=[]
    if len(coo)>0:
      coords.append(coo)

  print(spaces)
  return xy_rect, spaces

def get_image_segment(image, rect):
   return image[rect[0]:rect[1],rect[2]:rect[3]]

   
image=Image.open(image_path)# input image location
image=np.asarray(image)

## plot processed images
fig = plt.figure()
ax1 = plt.subplot(2,2,1)
imgplot = plt.imshow(image)
plt.title("Original")

gray_image=grayscale(image) # rgb to grayscale conversion

ax1 = plt.subplot(2,2,2)
imgplot = plt.imshow(gray_image, cmap='grey')
plt.title("Greyscale")

binary_image = binary(gray_image)
ax2 = plt.subplot(2,2,3)
imgplot = plt.imshow(binary_image, cmap='grey')
plt.title("binary")


# plot segmented imaged for visualization
fig2 = plt.figure()

ax4 = plt.subplot(1,2,2)
imgplot = plt.imshow(binary_image, cmap='grey')
plt.title("characters")

patch = []
rectangles = line_segment(binary_image)

for rect in rectangles:
  chars, spaces = char_segment(rect, binary_image)
  char_patches = []
  for char in chars:
     char_patches.append(Rectangle([char[2], char[0]], char[3]-char[2], char[1]-char[0], fill=False))
  
  collect1 = PatchCollection(char_patches, facecolor='none', ec='red', linewidth=1)
  for space in spaces:
    ax4.add_line(Line2D([space, space], [chars[0][0], chars[0][1]], color='blue'))
  ax4.add_collection(collect1)
  patch.append(Rectangle([rect[2], rect[0]], rect[3]-rect[2], rect[1]-rect[0], fill=False))

ax3 = plt.subplot(1,2,1)
imgplot = plt.imshow(binary_image, cmap='grey')
plt.title("lines")
collect = PatchCollection(patch, facecolor='none', ec='red', linewidth=1)
ax3.add_collection(collect)


plt.show()