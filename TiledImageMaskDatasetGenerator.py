# Copyright 2024 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 2024/08/04 sarah@antillia.com

import os
import shutil
import glob
import cv2

from PIL import Image
import numpy as np

import traceback

class TiledImageMaskDatasetGenerator:
  def __init__(self, exclude_empty=False, split_size=512):
    self.exclude_empty= exclude_empty
    self.split_size   = 512
    # The size of the original images and mask.
    self.WIDTH  = 1360
    self.HEIGHT = 1024
    self.RESIZED_WIDTH  = self.split_size * 3
    self.RESIZED_HEIGHT = self.split_size * 2
    print(" RESIZED: WIDTH {}  HEIGHT {}".format(self.RESIZED_WIDTH, self.RESIZED_HEIGHT))
    input("---------OK? ")

  def split(self, root_dir, output_dir):
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

      
    input_images_dir = root_dir + "/images/"
    input_masks_dir  = root_dir + "/masks/"

    output_images_dir = os.path.join(output_dir, "images")
    output_masks_dir  =  os.path.join(output_dir, "masks")
    if not os.path.exists(output_images_dir):
      os.makedirs(output_images_dir)
    if not os.path.exists(output_masks_dir):
      os.makedirs(output_masks_dir)
    
    self.split_one(input_masks_dir,  output_masks_dir, output_images_dir, mask=True)
    self.split_one(input_images_dir, output_masks_dir, output_images_dir, mask=False)


  def split_one(self, input_images_dir, output_masks_dir, output_images_dir,  mask=False):
    image_files  = glob.glob(input_images_dir + "/*.jpg")
    image_files += glob.glob(input_images_dir + "/*.png")
    image_files  = sorted(image_files)
    print("--- image_files {}".format(image_files))
    index = 10000
    split_size = self.split_size

    for image_file in image_files:
      index += 1

      image = Image.open(image_file)

      w, h  = image.size
      # 1 Resized image to 512x512 from 1360x1024
      # Don't care aspect-ratio in this resizing.
      resized = image.resize((self.split_size, self.split_size))
      if mask:
        output_mask_filepath  = os.path.join(output_masks_dir,  str(index) + ".jpg") 
        resized.save(output_mask_filepath)
      else:
        output_image_filepath = os.path.join(output_images_dir, str(index) + ".jpg") 
        resized.save(output_image_filepath)

      # Exapand image and masks      
      expanded_image = image.resize((self.RESIZED_WIDTH, self.RESIZED_HEIGHT))   
      #expanded_masks = mask.resize((self.RESIZED_WIDTH, self.RESIZED_HEIGHT))   

      # 2 Split 1024x1024 image to 512x512 tiles.

      vert_split_num  = self.RESIZED_HEIGHT // split_size
      #if h % split_size != 0:
      #  vert_split_num += 1

      horiz_split_num = self.RESIZED_WIDTH // split_size

      for j in range(vert_split_num):
        for i in range(horiz_split_num):
          left  = split_size * i
          upper = split_size * j
          right = left  + split_size
          lower = upper + split_size
          
 
          cropbox = (left,  upper, right, lower )
          
          # Crop a region specified by the cropbox from the whole image to create a tiled image segmentation.      
          cropped = expanded_image.crop(cropbox)

          #line = "image file {}x{} : x:{} y:{} width: {} height:{}\n".format(j, i, left, upper, cw, ch)
          #print(line)            
          cropped_image_filename = str(index) + "_" + str(j) + "x" + str(i) + ".jpg"
          output_mask_filepath  = os.path.join(output_masks_dir,  cropped_image_filename) 
          output_image_filepath = os.path.join(output_images_dir, cropped_image_filename) 

          if mask:
            cropped.save(output_mask_filepath)
            print("--- Saved {}".format(output_mask_filepath))
          else:
            cropped.save(output_image_filepath)
            print("--- Saved {}".format(output_image_filepath))

  def pil2cv(self, image):
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2: 
        pass
    elif new_image.shape[2] == 3: 
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4: 
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

if __name__ == "__main__":
  try:
    input_dir  = "./BreCaHAD-master"
    output_dir = "./Tiled-BreCaHAD-master"

    # Create Mixed (NonTiled and Tiled) MoNuSeg master from MoNuSeg-master-1024x1024 
    splitter = TiledImageMaskDatasetGenerator()

    splitter.split(input_dir, output_dir)
  except:
    traceback.print_exc()