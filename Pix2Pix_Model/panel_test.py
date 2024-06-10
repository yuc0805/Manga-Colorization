"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import torch
from tqdm import tqdm
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2
import torchvision.transforms.functional as F
from Manga_Panel_Extractor.panel_extractor import PanelExtractor
from skimage import io

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    #dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    panel_extractor = PanelExtractor(min_pct_panel=2, max_pct_panel=90)
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
  
    print('--------Start Inferencing!---------')
    test_folder_name = '/Users/leo/Desktop/pytorch-CycleGAN-and-pix2pix/colorization_dataset/gray_test'
    print('test_folder_name: ',test_folder_name)
    background_path = '/Users/leo/Desktop/Manga-Colorization/pix2pix_baseline_results/fake_B'

    load_background = True
    print('loading color background: ',str(load_background))
    print('--------Start Inferencing!---------')
    for img_name in tqdm(os.listdir(test_folder_name)):
        if img_name[0] == '.': continue
        img = os.path.join(test_folder_name,img_name)

        # get background image
        if load_background:
            original_img = io.imread(img)
            bg_h,bg_w= original_img.shape
            # Check the dimensions
            print('input shape for panel extractor',original_img.shape)
            # run the background image
            
            background_img_path = os.path.join(background_path,img_name) 
            image = io.imread(background_img_path)
            image = Image.fromarray(image).resize((bg_w, bg_h))
            background_img_path = os.path.join('/Users/leo/Desktop/Manga-Colorization/pix2pix_baseline_results/fake_B_dim', img_name)
            image.save(background_img_path)

        print('input for panel extractor',img)
        output_imgs_path,panels, masks, panel_masks = panel_extractor.extract(img)
        print(f'Start Coloring {len(panels)} images!')
        print('output_imgs_path: ',output_imgs_path[0])

        color_panels = []
        for image_path,panel_visual in tqdm(zip(output_imgs_path[0],panels)):
            #print(panel_visual)
            panel_visual = (panel_visual / 255.0) * 2 - 1
            panel_h,panel_w,_ = panel_visual.shape

            panel_visual = torch.tensor(panel_visual, dtype=torch.float32).permute(2, 0, 1) # 
            resize_to_input = transforms.Resize((256, 256))
            panel_visual = resize_to_input(panel_visual) # 3, 256, 256
            panel_visual = torch.mean(panel_visual, dim=0, keepdim=True)
            panel_visual = panel_visual.unsqueeze(0)

            panel_data_dict = {'A': panel_visual,
                                'B': panel_visual.expand(-1, 2, -1, -1),
                                'A_paths':[image_path],
                                'B_paths': [image_path]}
            
            image_name = os.path.basename(image_path)
            print('image path that are going to be color: ',image_name)

            model.set_input(panel_data_dict)  # unpack data from data loader
            model.test()           # run inference
            visuals = model.get_current_visuals()  # image vector for a single panel
            img_path = model.get_image_paths()
            
            color_panel = visuals['fake_B_rgb'].astype(np.uint8) #  (256,256,3)
            # #print('visuals keys: ',visuals.keys())
            color_panel = torch.tensor(visuals['fake_B_rgb']).permute(2, 0, 1) # permute to (3,256,256) to apply transform
            # #print('ouput shape of color_panel',color_panel.shape) # 3, 256 256
            color_panel = F.resize(color_panel, (panel_h, panel_w))
            # print('ouput shape of color_panel',color_panel.shape)
            color_panel = color_panel.permute(1, 2, 0).numpy().astype(np.uint8) # h, w,3
            
            print('shape of color_panel: ',color_panel.shape)

            color_panels.append(color_panel)

            output_path = os.path.join('colored_panels_output', image_name)
            im = Image.fromarray(color_panel)
            im.save(output_path)
            print('color pannel image save to: ',output_path)
            print(f'processing image...  {img_path}')

        print('len of color panels: ',len(color_panels))
        #print(color_panels)
        panel_extractor.concatPanels(background_img_path, color_panels, masks, panel_masks,
                                     out_folder='/Users/leo/Desktop/Manga-Colorization/panel_pix2pix_result')
        

# python panel_test.py --dataroot /Users/leo/Desktop/pytorch-CycleGAN-and-pix2pix/colorization_dataset --name color_pix2pix --model colorization --gpu_ids -1 --dataset_mode colorization
