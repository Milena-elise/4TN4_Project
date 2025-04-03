import numpy as np
import cv2 as cv
from skimage import io
import os
from PIL import Image
from PIL import ImageOps
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.collections import PatchCollection
import tensorflow as tf


image_path = 'images/PandP_C1/P1.png' #path to image
model = tf.keras.models.load_model('models/charCNN.keras') #path to model

def grayscale(a):
  '''
  Convert RGB image to grey
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
  Convert greyscale to binary image
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

def erosion(a, size):
  erod_sse = np.ones((3,3))
  
  mn = np.shape(a)
  m = mn[0]
  n = mn[1]
  g = np.zeros((m,n))
  a = a/255

  for i in range(m):
    for j in range(n):
      sum = 0
      for row in range(size):
        for col in range(size):
          p = int(i-(size-1)/2-1+row)
          q = int(j-(size-1)/2-1+col)
                    
          if p<=0 or p>m or q<=0 or q>n: # out of image bounds
            continue
                   
          else:
            sum =sum+a[p,q]*erod_sse[row,col]
  
          if sum == size*size:
            g[i,j] = 255 # assign to output val
  
  return g

def dilation(a, size):
  erod_sse = np.ones((3,3))
  
  mn = np.shape(a)
  m = mn[0]
  n = mn[1]
  g = np.zeros((m,n))
  a = a/255

  for i in range(m):
    for j in range(n):
      sum = 0
      for row in range(size):
        for col in range(size):
          p = int(i-(size-1)/2-1+row)
          q = int(j-(size-1)/2-1+col)
                    
          if p<=0 or p>m or q<=0 or q>n: # out of image bounds
            continue
                   
          else:
            sum =sum+a[p,q]*erod_sse[row,col]
  
          if sum != 0:
            g[i,j] = 255 # assign to output val
  
  return g

def line_coords(coords):
    '''
    Isolate bounding coordinates of line segments
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
    Isolate bounding cooridnates of characters within a line semgment
    coords - cooridinate of edges in character
    returns character boundry rectangle
    '''
    xmin=coords[0][0][1]
    xmax=coords[-1][0][1]
    ymin=20000
    ymax=0
    for i in coords:
        for j in i:
            if j[0] > ymax:
                ymax=j[0]
            if j[0] < ymin:
                ymin=j[0]

    return [line[0]+ymin-1,line[0]+ymax+1, line[2]+xmin,line[2]+xmax+2]

def line_segment(a):
  '''
  Find all in segments
  a - nparray of binary image
  returns - boundaries of each line

  function adapted from: https://medium.com/@magodiasanket/ocr-optical-character-recognition-from-scratch-using-deep-learning-a6a599963d71
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
  Find all characters in a line segment

  line - boundary of line
  image - nparray of binary image
  returns - boundaries of each character
  '''

  # funtion needs to have better transition between letter - some letter are close enough to get grouped together
  
  coords=[] #coordinates with text, boundaries
  xy_rect = [] # line detection info
  spaces = [] # for tracking location of spaces
  space_size = [] # for tracking lengths of gaps between letters
  a = image[line[0]:line[1],line[2]:line[3]+3]# added 3 so that last character was able to be detected 
  
  space_count = 0

  first_row = -1 # coordinate of fist non-empty row in column
  last_row = -1 # cooridate of last empy row in column
  first_row_prev = -1 # coordinate of first non-empy row in previos column
  last_row_prev = -1 # coordinate of late non-empy row in previous column
  
  char_width = 0 # track width of current character
  char_width_sum = 0 # to be used for average charaxcter width
  prev_char_width = 0 # width of previous character

  for i in range(len(a[0,:])): #for each col in image
    coo=[] # track locations of col pixels
    flag=0 # flag if column has text

    count = 0 # track row number
    col_gap = 0 # track length of empty space in column
    col_gap_max = 0 
    
    for c in a[:,i]: # iterate through all rows in column

      if c<200: # check row has text
       
        # update and reset cloumn gap tracker
        if col_gap > col_gap_max:
           col_gap_max = col_gap
        col_gap = 0
       
        # update first row
        if first_row == -1:
          first_row = count
          last_row = count
        
        # update last row
        else:
          last_row = count

        #add space, rest counter
        spaces.append(i+line[2]-space_count/2)
        space_size.append(space_count)
        space_count = 0
        
        flag=1
      
      # empty row in column
      else:
          col_gap += 1

      # 
      count +=1 # next row

    letter_thresh=0.2 # propertion of full line wideth needed to classify as character

    if flag==1: # col had text somewhere
      
      char_width += 1
      
      # check for letters very close together
      if ((first_row > last_row_prev and first_row > first_row_prev) or (last_row < last_row_prev and last_row < first_row_prev)): # still needs to seperate My and ay
 
          if len(coords)> 0: # end line
              xy_rect.append(char_coords(coords, line))
              coords=[]
              if char_width > prev_char_width:
                prev_char_width = char_width # avoid thin letters
              char_width_sum += char_width
              char_width = 0
              space_count += 1
      try:
        if a[first_row][i] <200:  # if text in the row to the below
          coo.append([first_row, i]) # add to list of edge changed
      except:
        pass

      try:
        if a[last_row][i] <200:  # if text in the row to the below
          coo.append([last_row, i]) # add to list of edge changed
      except:
        pass
        
  
    else: # col had no text
      space_count +=1
      
      if len(coords)> 0: # end line
          xy_rect.append(char_coords(coords, line))
          coords=[]

          # for calculating average char width
          if char_width > prev_char_width:
            prev_char_width = char_width 
          char_width_sum += char_width
          char_width = 0

    # update for next column
    first_row_prev = first_row
    last_row_prev = last_row
    first_row = -1
    last_row = -1
    col_gap_max = 0

    if len(coo)>0: # add column coords to character cords
      coords.append(coo)

  # average width
  try: av_char = char_width_sum/len(xy_rect)
  except:
     av_char = 0
  
  # decide if spaces are between letters in owrd or between words
  true_spaces = []
  for i in range(len(space_size)):
     if space_size[i] > 0.6*av_char:
        true_spaces.append(spaces[i])

  return xy_rect, true_spaces

def upsample_image(image, factor):
    '''
    Convert char image from to new larger size
    image - original image
    factor - what to scale image by
    returns - new image of desired size
    '''
    mn = np.shape(image)
    m = mn[0]
    n = mn[1]
    new_image = np.zeros((m*factor, n))
    for row in range(m):
       for i in range(factor):
        new_image[factor*row+i, :] = image[row, :]
    
    image = new_image
    new_image = np.zeros((m*factor, n*factor))
    for col in range(n):
       for i in range(factor):
        new_image[:, factor*col+i] = image[:, col]

          
    return new_image

def downsample_image(image, factor):
    '''
    Convert char image to new smaller size
    image - original image
    factor - what to scale image down by
    returns - new image of desired size
    '''
    mn = np.shape(image)
    m = mn[0]
    n = mn[1]
    down_m = int(m/factor)
    down_n = int(n/factor)
    new_image = np.zeros((down_m, n))
    for row in range(int(down_m)):
        new_image[row, :] = image[row*factor, :]
    
    image = new_image
    new_image = np.zeros((down_m, down_n))
    for col in range(int(down_n)):
        new_image[:, col] = image[:, col*factor]

    #print(np.shape(new_image))
    return new_image

def char_seg_2_28(image, char):
    '''
    converts character order info to 28 by 28 pixel image

    image - original image
    char - border of character (rectangle)
    '''

    out_size = 28
    border_min = 1 # empty pixels along edges
    char_im = get_image_segment(image, char)
    mn = np.shape(char_im)
    m = mn[0]
    n = mn[1]

    if m < out_size-border_min and n < out_size-border_min:
      upsample = int(np.floor(min((out_size-border_min)/m, (out_size-border_min)/n)))
      char_im = upsample_image(char_im, upsample)
    elif m > out_size-border_min or n > out_size-border_min:
      downsample = int(np.ceil(max(m/(out_size-border_min), n/(out_size-border_min))))
      char_im = downsample_image(char_im, downsample)
    
    mn = np.shape(char_im)
    m = mn[0]
    n = mn[1]
    
    top = int(np.ceil((out_size-m)/2))
    bottom = int(np.floor((out_size-m)/2))
    left = int(np.ceil((out_size-n)/2))
    right = int(np.floor((out_size-n)/2))

    new_im = 255*np.ones((out_size,out_size))

    new_im[top:(out_size-bottom), left:(out_size-right)] = char_im
    
    return new_im


def get_image_segment(image, rect):
   '''
   rectanglue border to image

   image - original image
   rect - border of segment
   '''
   return image[rect[0]:rect[1],rect[2]:rect[3]]

def plot_segments(binary_image, orig_image):
  '''
  Segment characters
  '''

  # character segment plot
  fig2 = plt.figure()
  ax4 = plt.subplot(1,2,2)
  imgplot = plt.imshow(binary_image, cmap='grey')
  plt.title("characters")

  
  line_patches = [] # for overlaying line detection
  text_detected = [] # string of text found
  
  line_borders = line_segment(binary_image) # border of all inie
 
  for line_rect in line_borders:
    char_borders, spaces = char_segment(line_rect, binary_image)
    
    char_patches = [] # for plotting character segments
    space_i = 0 # for iterating through spaces in line
    
    char_prev = char_borders[0][2]# end of previos character
    
    for char in char_borders:
      

      # add in spaces where applicable
      if space_i != -1 and spaces and (spaces[space_i] <= char[2] and spaces[space_i] > char_prev):

        text_detected.append(' ')
        space_i += 1
        if space_i >= len(spaces):
          space_i = -1
      
      char_prev = char[3] # update prevous character end
      char_patches.append(Rectangle([char[2], char[0]], char[3]-char[2], char[1]-char[0], fill=False))

      char_im = char_seg_2_28(orig_image, char) # get image to input to model
      prediction = model.predict(char_im.reshape(-1, 28,28,1)/255.0)
      c = np.argmax(prediction) # label assigned to detected character

      # conver to ascii value
      if (c>=0 and c<=9): ## numbers 0-9"
        new_c = c+48

      elif(c>=10 and c<=35): # upercase letters
        new_c = c-10+65

      elif(c>=36 and c<=61): # lowercase letters
        new_c = c-36+97

      # uncomment if you want to see character image
      '''
      img = plt.imshow(char_im, cmap='grey')
      plt.title(f'The result is likely: {chr(new_c)}')
      plt.show()'
      '''

      # add detected character
      text_detected.append(chr(new_c))

    # end of line
    text_detected.append('\n')

    char_collection = PatchCollection(char_patches, facecolor='none', ec='red', linewidth=1) # for plotting space locations
    
    # add character ans space collections to plot
    for space in spaces:
      ax4.add_line(Line2D([space, space], [line_rect[0], line_rect[1]], color='blue'))
    ax4.add_collection(char_collection)
    
    line_patches.append(Rectangle([line_rect[2], line_rect[0]], line_rect[3]-line_rect[2], line_rect[1]-line_rect[0], fill=False))
  
  # print output text
  print(''.join(text_detected))
  # tode dump this to .txt file
  
  # plot image with line segments
  ax3 = plt.subplot(1,2,1)
  imgplot = plt.imshow(binary_image, cmap='grey')
  plt.title("lines")
  line_collection = PatchCollection(line_patches, facecolor='none', ec='red', linewidth=1)
  ax3.add_collection(line_collection)
  #plt.show()
  

# main code
image=Image.open(image_path)# input image location
image=np.asarray(image)


# plot processed images
fig = plt.figure()
ax1 = plt.subplot(1,2,1)
imgplot = plt.imshow(image)
plt.title("Original")

gray_image=grayscale(image) # rgb to grayscale conversion


binary_image = binary(gray_image)
erod_image = erosion(binary_image, 1)

ax1 = plt.subplot(1,2,2)
imgplot = plt.imshow(binary_image, cmap='grey')
plt.title("binary")

plot_segments(binary_image, gray_image)
