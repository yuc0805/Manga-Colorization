# stdlib
import argparse
from argparse import RawTextHelpFormatter
# project
from panel_extractor import PanelExtractor

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
    
    print('--------Start Inferencing!---------')
    for img_name in tqdm(os.listdir(args.folder)):
        if img_name[0] == '.': continue
        img = os.path.join(args.folder,img_name)
        print('input for panel extractor',img)
        output_imgs_path,panels, masks, panel_masks = panel_extractor.extract(img)
         # list of list, each sublist contains all path to each sub-images
        print(f'Start Coloring {len(panels)} images!')
        print('output_imgs_path: ',output_imgs_path)
        color_panels = []
        for image_path in tqdm(output_imgs_path[0]):
            image_name = os.path.basename(image_path)
            print('image path that are going to be color: ',image_path)
            translated_image = sample_img2img_model(model,image_path,target_domain='colored')
            translated_image = translated_image[0, [2, 1, 0]] # [3, 368, 256]
            translated_image = (translated_image + 1) / 2 * 255
            
            #resize = transforms.Resize((256,256))
            #translated_image = resize(new_image)

            translated_image = translated_image.permute(1,2,0).numpy().astype(np.uint8)
            color_panels.append(translated_image)
            output_path = os.path.join('colored_panels_output', image_name)
            im = Image.fromarray(translated_image)
            im.save(output_path)
            print('color pannel image save to: ',output_path)
        
        # TODO: Concat
        panel_extractor.concatPanels(img, color_panels, masks, panel_masks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Implementation of a Manga Panel Extractor and dialogue bubble text eraser.",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument("-kt", "--keep_text", action='store_true',
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
