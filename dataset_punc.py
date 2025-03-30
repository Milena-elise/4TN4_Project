import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# Character mapping
class_to_char = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 
                 '.', ',', "'", ';', '!', '?']

# Configuration
CSV_PATH = 'data/typedCSV.csv'
OUTPUT_DIR = "data/dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# First, remove rows after 62993 if needed
df = pd.read_csv(CSV_PATH)
if len(df) > 62993:
    df = df.iloc[:62994]
    df.to_csv(CSV_PATH, index=False)
    print(f"Trimmed CSV to {len(df)} rows")

# Character definitions - now including both punctuation AND our target chars
CHARACTER_DATA = {
    # Punctuation
    '.': {'name': 'period', 'label': 62},
    ',': {'name': 'comma', 'label': 63},
    "'": {'name': 'quote', 'label': 64},
    ';': {'name': 'semicolon', 'label': 65},
    '!': {'name': 'exclamation', 'label': 66},
    '?': {'name': 'question', 'label': 67},
    # Special characters we want to enhance
    # 'i': {'name': 'lowercase_i', 'label': class_to_char.index('i')},
    '0': {'name': 'zero', 'label': class_to_char.index('0')},
    # 'o': {'name': 'lowercase_o', 'label': class_to_char.index('o')},
    'r': {'name': 'lowercase_r', 'label': class_to_char.index('r')},
    't': {'name': 'lowercase_t', 'label': class_to_char.index('t')},
    'j': {'name': 'lowercase_j', 'label': class_to_char.index('j')}
}

FONT_DIR = "/Library/Fonts/"
FONT_SIZES = [20, 30, 40, 50]

def get_font_paths():
    """Get all available font paths"""
    extensions = ('.ttf', '.otf', '.ttc')
    return [os.path.join(root, f) 
            for root, _, files in os.walk(FONT_DIR) 
            for f in files 
            if f.lower().endswith(extensions)]

def generate_char_image(char, font, size=30):
    """Generate image with the character"""
    img = Image.new('L', (100, 100), 255)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), char, font=font)
    x = (100 - (bbox[2]-bbox[0])) // 2 - bbox[0]
    y = (100 - (bbox[3]-bbox[1])) // 2 - bbox[1]
    draw.text((x, y), char, font=font, fill=0)
    return img

def create_char_variations(image, char, target_size=28):
    """Create multiple scaled versions without special modifications for 0 and o"""
    img_array = np.array(image)
    rows = np.any(img_array < 255, axis=1)
    cols = np.any(img_array < 255, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return []
    
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    
    # Standard padding for all characters
    padding = 4
    vertical_shift = 0
    
    ymin_pad = max(0, ymin-padding)
    ymax_pad = min(image.height, ymax+padding)
    xmin_pad = max(0, xmin-padding)
    xmax_pad = min(image.width, xmax+padding)
    
    char_img = image.crop((xmin_pad, ymin_pad, xmax_pad+1, ymax_pad+1))
    width, height = char_img.size
    
    variations = []
    for zoom in ['normal', 'zoomed', 'max']:
        if zoom == 'normal':
            scale = min((target_size-8)/width, (target_size-8)/height)
        elif zoom == 'zoomed':
            scale = min((target_size-4)/width, (target_size-4)/height)
        else:  # max zoom
            scale = min(target_size/width, target_size/height)
        
        new_w, new_h = int(width*scale), int(height*scale)
        resized = char_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        new_img = Image.new('L', (target_size, target_size), 255)
        paste_x = (target_size - new_w) // 2
        paste_y = (target_size - new_h) // 2 + vertical_shift
        new_img.paste(resized, (paste_x, paste_y))
        
        variations.append(new_img)
    
    return variations

def append_to_csv(csv_path, new_data):
    """Append new data to CSV maintaining the exact format"""
    columns = ['label'] + [f'pixel {i}' for i in range(28*28)]
    
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        new_df = pd.DataFrame(new_data, columns=columns)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = pd.DataFrame(new_data, columns=columns)
    
    combined_df.to_csv(csv_path, index=False)
    return len(new_data)

def generate_enhanced_dataset():
    """Generate enhanced samples for all target characters"""
    font_paths = get_font_paths()
    if not font_paths:
        print("No fonts found in directory:", FONT_DIR)
        return
    
    csv_data = []
    generated_samples = 0
    
    for char, data in CHARACTER_DATA.items():
        char_dir = os.path.join(OUTPUT_DIR, data['name'])
        os.makedirs(char_dir, exist_ok=True)
        
        for font_path in font_paths:
            for size in FONT_SIZES:
                try:
                    font = ImageFont.truetype(font_path, size)
                    # print(f"Processing {char} with {os.path.basename(font_path)} size {size}")
                    
                    img = generate_char_image(char, font, size)
                    variations = create_char_variations(img, char)
                    
                    for i, variation in enumerate(variations):
                        # Save image
                        font_name = os.path.splitext(os.path.basename(font_path))[0]
                        suffix = ['normal', 'zoomed', 'max'][i]
                        filename = f"{char_dir}/{font_name}_{size}_{suffix}.png"
                        success = cv2.imwrite(filename, np.array(variation))
                        
                        if success:
                            # Prepare CSV row
                            pixels = np.array(variation).flatten().tolist()
                            csv_data.append([data['label']] + pixels)
                            generated_samples += 1
                        else:
                            print(f"Failed to save {filename}")
                            
                except Exception as e:
                    print(f"Skipping {font_path} size {size}: {str(e)}")
                    continue
    
    if csv_data:
        count = append_to_csv(CSV_PATH, csv_data)
        print(f"\nAdded {count} new enhanced samples to {CSV_PATH}")
        print(f"Total generated samples: {generated_samples}")
        
        # Verification
        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH)
            print(f"\nCSV now contains {len(df)} total rows")
            print("Counts by character:")
            for char in CHARACTER_DATA:
                count = len(df[df['label'] == CHARACTER_DATA[char]['label']])
                print(f"{char} (label {CHARACTER_DATA[char]['label']}): {count} samples")
    else:
        print("No data was generated")

# Run the generator
generate_enhanced_dataset()