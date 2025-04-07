import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.collections import PatchCollection
import tensorflow as tf
from gtts import gTTS

import nltk
from nltk.corpus import words

# Download English words list if needed

nltk.download('words')

# Get English vocabulary set
english_words = set(words.words())

def is_real_word(word):
    """Check if a word exists in English dictionary (case-insensitive)"""
    return word.lower() in english_words

def correct_ending_o_or_0(text):
    """
    Converts ending 'o' or '0' to '.' only if it's not a real word.
    If '0' is at the end, it is first converted to 'o' before checking.
    Preserves all other text exactly.
    """
    paragraphs = text.split('\n')
    corrected = []
    
    for para in paragraphs:
        words = para.split()  # Split paragraph into words
        corrected_words = []
        
        for word in words:
            # Check if word ends with 'o' or '0'
            if word[-1].lower() == 'o' or word[-1] == '0':
                # If last character is '0', replace it with 'o' for validation
                if word[-1] == '0':
                    word = word[:-1] + 'o'
                
                # Check if it's a real word (excluding punctuation)
                clean_word = word.rstrip('.,;!?\'"')
                if not is_real_word(clean_word):
                    word = word[:-1] + '.'  # Replace final 'o' or '0' with '.'
            
            corrected_words.append(word)
        
        # Join words back into a corrected paragraph
        corrected.append(' '.join(corrected_words))
    
    return '\n'.join(corrected)


def correct_text(ocr_text):
    """
    Corrects OCR output with:
    1. Smart 0→o substitution (only within words)
    2. Converts ending 0 to period per paragraph
    3. ;→i substitution in words
    4. Punctuation and capitalization fixes
    """

    ocr_text = ocr_text.replace("''", '"')
    ocr_text = ocr_text.replace("'", ',')


    ocr_text = correct_ending_o_or_0(ocr_text)

    # Split into paragraphs
    paragraphs = ocr_text.split('\n')
    corrected_paragraphs = []
    
    for para in paragraphs:
        # Convert ending 0 to period if it's at paragraph end
        para = para.rstrip()
        if len(para) > 0 and para[-1] == '0':
            para = para[:-1] + '.'
            
        words = para.split()
        corrected_words = []
        
        for word in words:
            # Only modify if it's likely a word (contains letters)
            if any(c.isalpha() for c in word):
                # Fix 0→o when surrounded by letters
                if '0' in word:
                    new_word = []
                    for i, c in enumerate(word):
                        if c == '0' and ((i > 0 and word[i-1].isalpha()) or 
                                        (i < len(word)-1 and word[i+1].isalpha())):
                            new_word.append('o')
                        else:
                            new_word.append(c)
                    word = ''.join(new_word)
                
                # Fix ;→i in words
                if ';' in word:
                    word = word.replace(';', 'i')
            
            corrected_words.append(word)
        
        corrected_paragraphs.append(' '.join(corrected_words))
    
    # Rejoin paragraphs
    text = '\n'.join(corrected_paragraphs)
    
    # Handle punctuation and capitalization
    chars = list(text)
    sentence_start = True  # Indicates if the next word should start a sentence
    in_proper_noun = False  # Tracks if the current word is a proper noun
    
    for i in range(len(chars)):
        current = chars[i]
        prev_char = chars[i-1] if i > 0 else ' '
        
        # If it's a letter and we are at the start of a sentence, capitalize it
        if sentence_start and current.isalpha():
            chars[i] = current.upper()
            sentence_start = False  # After capitalizing, next word won't be a sentence start
            
        # If we are not at the start of the sentence, convert it to lowercase
        elif current.isalpha():
            chars[i] = current.lower()
            
        # Reset sentence_start when encountering punctuation marks (end of sentence).
        if current in ['.', '!', '?']:
            sentence_start = True
            # Skip any whitespace after punctuation
            j = i + 1
            while j < len(chars) and chars[j].isspace():
                j += 1
            # Capitalize the first letter of the next word (start of the sentence)
            if j < len(chars) and chars[j].isalpha():
                chars[j] = chars[j].upper()
                i = j
    
    text = ''.join(chars)

    return text

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


def binary(a, T=150):# takes np array image
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

def erosion(a, size1, size2):
  '''
  image erosion
  convolves images with size 1 by size 2 matrix
  a - input image
  '''
  erod_sse = np.ones((size1,size2))
  
  mn = np.shape(a)
  m = mn[0]
  n = mn[1]
  g = np.zeros((m,n))
  a = (255-a)/255

  for i in range(m):
    for j in range(n):
      sum = 0
      for row in range(size1):
        for col in range(size2):
          p = int(i-(size1-1)/2-1+row)
          q = int(j-(size2-1)/2-1+col)
                    
          if p<0 or p>=m or q<0 or q>=n: # out of image bounds
            continue
                   
          else:
            sum =sum+a[p,q]*erod_sse[row,col]
  
          if sum == size1*size2:
            g[i,j] = 255 # assign to output val
  
  return 255*np.ones((m,n))-g

def dilation(a, size1, size2):
  '''
  modified image dilation
  - result of convolution with sse must be at least size 1 (instead of 1)
  convolves images with size 1 by size 2 matrix
  a - image
  '''
  erod_sse = np.ones((size1,size2))
  
  mn = np.shape(a)
  m = mn[0]
  n = mn[1]
  g = np.zeros((m,n))
  a = (255-a)/255

  for i in range(m):
    for j in range(n):
      sum = 0
      for row in range(size1):
        for col in range(size2):
          p = int(i-(size1-1)/2-1+row)
          q = int(j-(size2-1)/2-1+col)
                    
          if p<0 or p>=m or q<0 or q>=n: # out of image bounds
            continue
                   
          else:
            sum =sum+a[p,q]*erod_sse[row,col]
  
          if sum >= size1:
            g[i,j] = 255 # assign to output val
  
  return 255*np.ones((m,n))-g

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
    ymin = min([j[0] for i in coords for j in i])
    ymax = max([j[0] for i in coords for j in i])
    xmin = min([j[1] for i in coords for j in i])
    xmax = max([j[1] for i in coords for j in i])

    # Ensure a minimum height and width for punctuation
    min_height = 8  # Adjust as needed
    min_width = 8  # Adjust as needed

    if ymax - ymin < min_height:
        ymin -= (min_height - (ymax - ymin)) // 2
        ymax += (min_height - (ymax - ymin)) // 2

    if xmax - xmin < min_width:
        xmin -= (min_width - (xmax - xmin)) // 2
        xmax += (min_width - (xmax - xmin)) // 2

    return [line[0] + ymin - 1, line[0] + ymax + 1, line[2] + xmin, line[2] + xmax + 2]

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
      '''
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
      '''
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

  #character segment plot
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
      prediction = model.predict(char_im.reshape(-1, 28,28,1)/255.0, verbose=0)
      c = np.argmax(prediction) # label assigned to detected character

      # conver to ascii value
      if (c>=0 and c<=9): ## numbers 0-9"
        new_c = c+48

      elif(c>=10 and c<=35): # upercase letters
        new_c = c-10+65

      elif(c>=36 and c<=61): # lowercase letters
        new_c = c-36+97

      elif(c==62): 
        new_c = ord('.')

      elif(c==63): 
        new_c = ord(',')

      elif(c==64): 
        new_c = ord("'")

      elif(c==65):
        new_c = ord(';')

      elif(c==66):
        new_c = ord('!')

      elif(c==67):
        new_c = ord('?')

    
      # uncomment if you want to see character image
      # img = plt.imshow(char_im, cmap='grey')
      # plt.title(f'The result is likely: {chr(new_c)}')
      # plt.show()


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
  output_txt = ''.join(text_detected)
  print(output_txt + '\n')

  output_txt = correct_text(output_txt)

  print(output_txt + '\n')


  output_path = 'output.txt'
  with open(output_path, 'w') as f:
      f.write(output_txt)
  print(f"Text saved to: {output_path}")
  
  # plot image with line segments
  ax3 = plt.subplot(1,2,1)
  imgplot = plt.imshow(binary_image, cmap='grey')
  plt.title("lines")
  line_collection = PatchCollection(line_patches, facecolor='none', ec='red', linewidth=1)
  ax3.add_collection(line_collection)
  plt.show()



def contextual_correction(char, position_in_word, word_position_in_line):
    """
    Handles ALL character corrections in one place:
    - Fixes universal misclassifications (e.g., '1' → 'l')
    - Applies word-aware fixes (e.g., '0' → 'o' only inside words)
    - Manages uppercase/lowercase rules
    """
 
    # Word-aware fixes
    if position_in_word > 0:  # Only modify if inside a word
        if char == '0': 
            char = 'o'  #'0' → 'o'
        elif char == ';':
            char = 'i'

    # Case handling
    if char.isupper():
        if position_in_word > 0:  # Lowercase if not word-start
            char = char.lower()
        elif word_position_in_line == 0:  # Keep uppercase if line-start (proper noun)
            pass
        else:  # Optional: lowercase standalone capitals
            char = char.lower()

    return char


def reconstruct_text_with_corrections(text_detected):
    """
    Reconstructs the final text with corrections:
    - Handles spaces and newlines properly
    - Applies contextual fixes
    """
    corrected_text = []
    current_word = []
    word_position = 0  # Track word position in line
    
    for char in text_detected:
        if char == ' ' or char == '\n':  # Word boundary
            # Process the completed word
            for i, c in enumerate(current_word):
                corrected_char = contextual_correction(c, i, word_position)
                corrected_text.append(corrected_char)
            
            corrected_text.append(char)
            current_word = []
            word_position += 1 if char == ' ' else 0  # New line resets word count
        else:
            current_word.append(char)
    
    return ''.join(corrected_text)








####################################################################### main code

image_path = 'images/PandP_typed.jpeg' #path to image
model = tf.keras.models.load_model('models/charCNN.keras') #path to model
image=Image.open(image_path)# input image location
image=np.asarray(image)


# plot processed images
fig = plt.figure()
ax1 = plt.subplot(1,2,1)
imgplot = plt.imshow(image)
plt.title("Original")

gray_image=grayscale(image) # rgb to grayscale conversion

binary_image = binary(gray_image, T=200)


ax1 = plt.subplot(1,2,2)
imgplot = plt.imshow(binary_image, cmap='grey')
plt.title("binary")

new_image = dilation(binary_image, 3,2)
new_image = erosion(new_image, 3,3)

ax1 = plt.subplot(1,2,2)
imgplot = plt.imshow(new_image, cmap='grey')
plt.title("Pre-processed")
plt.show()

plot_segments(new_image, gray_image)


# Read the saved text
with open('output.txt', "r", encoding="utf-8") as file:
    text = file.read()

# Convert text to speech
tts = gTTS(text=text, lang="en")
speech_file = "output_audio.mp3"
tts.save(speech_file)

# Play the audio file (this works on most systems)
# os.system(f"start {speech_file}")  # Windows
# os.system(f"afplay {speech_file}")  # macOS
# os.system(f"mpg321 {speech_file}")  # Linux









