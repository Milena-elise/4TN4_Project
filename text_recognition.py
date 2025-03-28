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
from pytesseract import pytesseract 
import tensorflow as tf
#import easyocr

image_path = 'images/PandP_C1/P9.png'
model = tf.keras.models.load_model('models/charCNN.keras')
#reader = easyocr.Reader(['en'], gpu=False, )

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
            if j[0] > ymax:
                ymax=j[0]
            if j[0] < ymin:
                ymin=j[0]

    return [line[0]+ymin-1,line[0]+ymax+1, line[2]+xmin,line[2]+xmax+2]

def line_segment(a):
  '''
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
  line - boundary of line
  image - nparray of binary image
  returns - boundaries of each character
  '''

  # funntion needs to have better transition between letter - some letter are close enough to get grouped together
  
  coords=[] #coordinates with text, on edge change
  xy_rect = [] # line detection info
  spaces = []
  space_size = []
  a = image[line[0]:line[1],line[2]:line[3]+3]# added 3 so that last character was detected - todo find better fix
  
  space_count = 0
  spaces = []
  first_row = -1
  last_row = -1
  first_row_prev = -1
  last_row_prev = -1
  char_width = 0
  char_width_sum = 0
  prev_char_width = 0
  for i in range(len(a[0,:])): #for each col in image
    coo=[]
    flag=0

    count = 0 # track col number
    col_gap = 0
    col_gap_max = 0
    prev_gap_max = 0
    
    for c in a[:,i]: # iterate through all rows in column

      if c<200: # check col has any text in it
        if col_gap > col_gap_max:
           col_gap_max = col_gap
        col_gap = 0
        if first_row == -1:
          first_row = count
          last_row = count
        else:
          last_row = count



        #if space_count > prev_char_width*0.3:
        spaces.append(i+line[2]-space_count/2)
        space_size.append(space_count)
        

        space_count = 0
        flag=1
      
      else:
          col_gap += 1

      count +=1
    '''
    if (abs(first_row - last_row) < 3) and (abs(first_row_prev - last_row_prev) > abs(first_row - last_row)): # avoid connecting artefacts
      flag = 0
    '''

    letter_thresh=0.2
    if flag==1: # col had text somewhere
      char_width += 1
      # check for letters very close together
      #(first_row != -1 and first_row_prev != -1) and abs(last_row_prev - first_row) > letter_thresh*len(a[:,i]) and abs(first_row_prev - last_row) > letter_thresh*len(a[:,i]) and abs(last_row_prev - first_row) > letter_thresh*len(a[:,i]) and abs(first_row_prev - last_row) > letter_thresh*len(a[:,i]) and abs(first_row_prev - first_row) > letter_thresh*len(a[:,i]) and abs(last_row_prev - last_row) > letter_thresh*len(a[:,i]) and''' 
      if ((first_row > last_row_prev and first_row > first_row_prev) or (last_row < last_row_prev and last_row < first_row_prev)): # still needs to spearate My and ay
 
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
      
      '''
      for b in range(len(a[:,i])): # look for edge changes
        
        # no
        if a[b][i]>200: # if no text in current row
          try:
            if a[b+1][i] <200:  # if text in the row to the below
              coo.append([b+1, i]) # add to list of edge changed
          except:
            pass

        if a[b][i]<200: #if text in current row
          try:
            if a[b+1][i] > 200: # if next row empty
              coo.append([b,i]) # add to list
              if b == 0:
                 print("0 row added")
    
          except:
            pass
        '''
        
  
    else: # col had no text
      space_count +=1
      

      if len(coords)> 0: # end line0
          char_rect = char_coords(coords, line)
          if abs(char_rect[0]-char_rect[1]) > letter_thresh*(line[0]-line[1]):
            xy_rect.append(char_coords(coords, line))
          coords=[]
          if char_width > prev_char_width:
            prev_char_width = char_width # avoid thin letters
          char_width_sum += char_width
          char_width = 0

    
    first_row_prev = first_row
    last_row_prev = last_row
    first_row = -1
    last_row = -1
    prev_gap_max = col_gap_max
    col_gap_max = 0

    if len(coo)>0:
      coords.append(coo)
  if(len(xy_rect) != 0):
    av_char = char_width_sum/len(xy_rect)
  else:
     av_char = 0
  true_spaces = []
  for i in range(len(space_size)):
     if space_size[i] > 0.6*av_char:
        #space_size = np.delete(space_size, i)
        true_spaces.append(spaces[i])

  return xy_rect, true_spaces

def upsample_image(image, factor):
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
    out_size = 28
    border_min = 1
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
   return image[rect[0]:rect[1],rect[2]:rect[3]]

def plot_segments(binary_image, orig_image):
  #plot segmented imaged for visualization
  fig2 = plt.figure()

  ax4 = plt.subplot(1,2,2)
  imgplot = plt.imshow(binary_image, cmap='grey')
  plt.title("characters")
  patch = []
  rectangles = line_segment(binary_image)


  #rectangles = rectangles[3]
  for rect in rectangles:
    #fig3 = plt.figure()
    #img = plt.imshow(get_image_segment(binary_image, rect), cmap='grey')
    chars, spaces = char_segment(rect, binary_image)
    char_patches = []
    for char in chars:
      char_patches.append(Rectangle([char[2], char[0]], char[3]-char[2], char[1]-char[0], fill=False))

      char_im = char_seg_2_28(orig_image, char)
      prediction = model.predict(char_im.reshape(-1, 28,28,1)/255.0)
      c = np.argmax(prediction)

      if (c>=0 and c<=9): ## numbers 0-9"
        new_c = c+48

      elif(c>=10 and c<=35): # upercase letters
        new_c = c-10+65

      elif(c>=36 and c<=61): # lowercase letters
        new_c = c-36+97


      img = plt.imshow(char_im, cmap='grey')
      plt.title(f'The result is likely: {chr(new_c)}')
      plt.show()


  
    collect1 = PatchCollection(char_patches, facecolor='none', ec='red', linewidth=1)
    for space in spaces:
      ax4.add_line(Line2D([space, space], [rect[0], rect[1]], color='blue'))
    ax4.add_collection(collect1)
    patch.append(Rectangle([rect[2], rect[0]], rect[3]-rect[2], rect[1]-rect[0], fill=False))

  ax3 = plt.subplot(1,2,1)
  imgplot = plt.imshow(binary_image, cmap='grey')
  plt.title("lines")
  collect = PatchCollection(patch, facecolor='none', ec='red', linewidth=1)
  ax3.add_collection(collect)
  plt.show()

from pdf2image import convert_from_path

# Store Pdf with convert_from_path function
#images = convert_from_path(image_path)
images = [1]

for i in range(len(images)):
      # Save pages as images in the pdf
    #images[i].save('page'+ str(i) +'.jpg', 'JPEG')
  
    image=Image.open(image_path)# input image location
    image=np.asarray(image)

    # plot processed images
    fig = plt.figure()
    ax1 = plt.subplot(2,2,1)
    imgplot = plt.imshow(image)
    plt.title("Original")

    gray_image=grayscale(image) # rgb to grayscale conversion

    ax1 = plt.subplot(2,2,2)
    imgplot = plt.imshow(gray_image, cmap='grey')
    plt.title("Greyscale")

    binary_image = binary(gray_image)
    # 

    plot_segments(binary_image, gray_image)