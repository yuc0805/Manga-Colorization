# stdlib
import argparse
from argparse import RawTextHelpFormatter
# project
from Manga_Panel_Extractor.panel_extractor import PanelExtractor
from skimage import io
from tqdm import tqdm

# cyclegan dependecies
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import mmcv
import mmgen
from mmgen.apis import init_model, sample_img2img_model

import time
from PIL import Image
import os
from torchvision.utils import save_image
from torchvision import transforms
import torchvision.transforms.functional as F

# pix2pix dependency
import os
from pix2pix.options.test_options import TestOptions
from pix2pix.models import create_model
#from pix2pix.data import create_dataset
from pix2pix.util.visualizer import save_images
from pix2pix.util import html
import torch

def main(args):
    panel_extractor = PanelExtractor(min_pct_panel=args.min_panel, max_pct_panel=args.max_panel)
    
    # Step1. Extract the pannels and removing text,
    # Save the pannels image
    # Record the path
    
    #output_imgs_path = panel_extractor.extract(args.folder) 
   
    #print('panels shape',panels[0].shape)

    #print('Number of images that need to be color',len(output_imgs_path))

    # TODO: colorize each sublist, and concat back to a full image
    config_file = '/Users/leo/Desktop/Manga-Colorization/Manga_CycleGAN/cyclegan_lsgan_resnet_in_summer2winter_b1x1_250k.py'
    checkpoint_file = '/Users/leo/Desktop/Manga-Colorization/Manga_CycleGAN/weight/iter_160000.pth'
    model = init_model(config_file, checkpoint_file, device=args.device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model = %s" % str(model))
    print('number of params : %.2f M' % (n_parameters%1e6))
    
    # Step 2: Use CycleGAN to color
    load_color_background = True
    print('loading color background: ',str(load_color_background))
    print('--------Start Inferencing!---------')
    # run the background image
    background_path = '/Users/leo/Desktop/Manga-Colorization/colored_background'
    for img_name in tqdm(os.listdir(args.folder)):
        if img_name[0] == '.': continue
        img = os.path.join(args.folder,img_name)

        if load_color_background:
            original_img = io.imread(img)
            bg_h,bg_w= original_img.shape
            # Check the dimensions
            print('input shape for panel extractor',original_img.shape)
            # # run the background image
            background_img = sample_img2img_model(model,img,target_domain='colored')
            background_img = background_img[0, [2, 1, 0]] # [3, 368, 256]
            background_img = (background_img + 1) / 2 * 255
            background_img =  F.resize(background_img, (bg_h, bg_w))
            background_img = background_img.permute(1,2,0).numpy().astype(np.uint8)
            im = Image.fromarray(background_img)
            background_output_path = os.path.join(background_path,img_name)
            im.save(background_output_path)
            print('color background image save to: ',background_output_path)

        # get list of panels
        output_imgs_path,panels, masks, panel_masks = panel_extractor.extract(img)
        print(f'Start Coloring {len(panels)} images!')
        print('output_imgs_path: ',output_imgs_path)
        color_panels = []
        for image_path,panel in zip(tqdm(output_imgs_path[0]),panels):
            print('panels shape',panel.shape)
            panel_h,panel_w,_ = panel.shape
            
            image_name = os.path.basename(image_path)
            print('image path that are going to be color: ',image_path)
            translated_image = sample_img2img_model(model,image_path,target_domain='colored')
            translated_image = translated_image[0, [2, 1, 0]] # [3, 368, 256]
            translated_image = (translated_image + 1) / 2 * 255
            translated_image =  F.resize(translated_image, (panel_h, panel_w))
            translated_image = translated_image.permute(1,2,0).numpy().astype(np.uint8)
            print('append color panel with size',translated_image.shape)

            color_panels.append(translated_image)
            output_path = os.path.join('cyclegan_colored_panels_output', image_name)
            im = Image.fromarray(translated_image)
            im.save(output_path)
            print('color pannel image save to: ',output_path)
        
        # TODO: Concat
        concatPanels_path = background_output_path if load_color_background else img
        panel_extractor.concatPanels(concatPanels_path, color_panels, masks, panel_masks,
                                     out_folder='/Users/leo/Desktop/Manga-Colorization/panel_cyclegan_result')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Implementation of a Manga Panel Extractor and dialogue bubble text eraser.",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument("-kt", "--keep_text", default=False,
                        help="Do not erase the dialogue bubble text.")
    parser.add_argument("-minp", "--min_panel", type=int, choices=range(1, 99), default=2, metavar="[1-99]",
                        help="Percentage of minimum panel area in relation to total page area.")
    parser.add_argument("-maxp", "--max_panel", type=int, choices=range(1, 99), default=90, metavar="[1-99]",
                        help="Percentage of minimum panel area in relation to total page area.")
    parser.add_argument("-f", '--folder', default='./images/', type=str,
                        help="""folder path to input manga pages.
Panels will be saved to a directory named `panels` in this folder.""")
    
    #  
    parser.add_argument('--device', default='cpu',
                        help='device to use for training / testing')

    args = parser.parse_args()
    main(args)

# python -m colorize_main --folder /Users/leo/Desktop/pytorch-CycleGAN-and-pix2pix/colorization_dataset/gray_test 
# python -m colorize_main --folder /Users/leo/Desktop/Manga-Panel-Extractor-master/Test
 