# stdlib
from os.path import splitext, basename, exists, join
from os import makedirs
import os
# 3p
from tqdm import tqdm
import numpy as np
from skimage import measure
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import cv2
# project
from text_detector.main_text_detector import TextDetector
from utils import get_files, load_image
from skimage import io


class PanelExtractor:
    def __init__(self, keep_text=False, min_pct_panel=2, max_pct_panel=90, paper_th=0.35):
        self.keep_text = keep_text
        assert min_pct_panel < max_pct_panel, "Minimum percentage must be smaller than maximum percentage"
        self.min_panel = min_pct_panel / 100
        self.max_panel = max_pct_panel / 100
        self.paper_th = paper_th

        # Load text detector
        print('Load text detector ... ', end="")
        self.text_detector = TextDetector()
        print("Done!")

    def _generate_panel_blocks(self, img):
        img = img if len(img.shape) == 2 else img[:, :, 0]
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        thresh = cv2.threshold(blur, 230, 255, cv2.THRESH_BINARY)[1]
        cv2.rectangle(thresh, (0, 0), tuple(img.shape[::-1]), (0, 0, 0), 10)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
        ind = np.argsort(stats[:, 4], )[::-1][1]
        panel_block_mask = ((labels == ind) * 255).astype("uint8")
        return panel_block_mask

    def generate_panels(self, img):
        block_mask = self._generate_panel_blocks(img)
        cv2.rectangle(block_mask, (0, 0), tuple(block_mask.shape[::-1]), (255, 255, 255), 10)

        # detect contours
        contours, hierarchy = cv2.findContours(block_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        panels = []
        masks = []
        panel_masks = []

        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            img_area = img.shape[0] * img.shape[1]

            # if the contour is very small or very big, it's likely wrongly detected
            if area < (self.min_panel * img_area) or area > (self.max_panel * img_area):
                continue

            x, y, w, h = cv2.boundingRect(contours[i])
            masks.append(cv2.boundingRect(contours[i]))
            # create panel mask
            panel_mask = np.ones_like(block_mask, "int32")
            cv2.fillPoly(panel_mask, [contours[i].astype("int32")], color=(0, 0, 0))
            panel_mask = panel_mask[y:y+h, x:x+w].copy()
            # apply panel mask
            panel = img[y:y+h, x:x+w].copy()
            panel[panel_mask == 1] = 255
            panels.append(panel)
            panel_masks.append(panel_mask)

        return panels, masks, panel_masks

    def remove_text(self, imgs):
        # detect text
        res = self.text_detector.detect(imgs)

        print("Removing text ... ", end="")
        text_masks = []
        for i, (_, polys) in enumerate(res):
            mask_text = np.zeros_like(imgs[i], "int32")
            for poly in polys:
                cv2.fillPoly(mask_text, [poly.astype("int32")], color=(255, 255, 255))
            text_masks.append(mask_text)

        # self.get_speech_bubble_mask(imgs, text_masks)

        without_text = []
        for i, img in enumerate(imgs):
            img[text_masks[i] == 255] = 255
            without_text.append(img)
        print("Done!")

        return without_text

    def get_speech_bubble_mask(self, imgs, text_masks):
        bubble_masks = []
        for i, img in enumerate(imgs):
            # get connected components
            _, bw = cv2.threshold(img, 230, 255.0, cv2.THRESH_BINARY)
            all_labels = measure.label(bw, background=25000)

            # speech labels
            labels = np.unique(all_labels[text_masks[i] == 255])

            # buble mask
            bubble_masks.append(np.isin(all_labels, labels) * 255)

    def extract(self, folder):
        print("Loading images ... ", end="")
        image_list, _, _ = get_files(folder)
        folder_file = join(folder, "panels")
        # image_list = []
        # image_list.append(folder)
        # print(image_list)
        imgs = [load_image(x) for x in image_list]
        print("Done!")

        # create panels dir
        if not exists(join(folder, "panels")):
            makedirs(join(folder, "panels"))
        folder = join(folder, "panels")

        # remove images with paper texture, not well segmented
        paperless_imgs = []
        for img in tqdm(imgs, desc="Removing images with paper texture"):
            hist, bins = np.histogram(img.copy().ravel(), 256, [0, 256])
            if np.sum(hist[50:200]) / np.sum(hist) < self.paper_th:
                paperless_imgs.append(img)

        # remove text from panels
        if not self.keep_text:
            paperless_imgs = self.remove_text(paperless_imgs)
        
        if not paperless_imgs:
            return imgs, [], []
        # print("can I print?")
        for i, img in tqdm(enumerate(paperless_imgs), desc="extracting panels"):
            panels, masks, panel_masks = self.generate_panels(img)
            name, ext = splitext(basename(image_list[i]))
            for j, panel in enumerate(panels):
                panel = np.array(panel)
                cv2.imwrite(join(folder, f'{name}_{j}{ext}'), panel)
            # show the order of colorized panels
            # img = Image.fromarray(img)
            # draw = ImageDraw.Draw(img)
            # font = ImageFont.truetype('./Open-Sans-Bold.ttf', 160)

            # def flatten(l):
            #     for el in l:
            #         if isinstance(el, list):
            #             yield from flatten(el)
            #         else:
            #             yield el

            # for i, bbox in enumerate(flatten(masks), start=1):
            #     w, h = draw.textsize(str(i), font=font)
            #     y = (bbox[1] + bbox[3] / 2 - h / 2)
            #     x = (bbox[0] + bbox[2] / 2 - w / 2)
            #     draw.text((x, y), str(i), (255, 215, 0), font=font)
            # img.show()
            return panels, masks, panel_masks

    def concatPanels(self, img_file, fake_imgs, masks, panel_masks):


        image_list, _, _ = get_files(img_file)
        print("hi this is image list", image_list)
        img = Image.open(image_list[0])
        img = np.array(img)


        fake_imgs = [load_image(x) for x in image_list[1:]]
        # print(fake_imgs)
        # out_imgs.append(f"D:\MyProject\Python\DL_learning\Manga-Panel-Extractor-master\out\in0_ref0.png")
        # out_imgs.append(f"D:\MyProject\Python\DL_learning\Manga-Panel-Extractor-master\out\in1_ref1.png")
        # out_imgs.append(f"D:\MyProject\Python\DL_learning\Manga-Panel-Extractor-master\out\in2_ref2.png")
        print("len of fake img", len(fake_imgs))
        mask_used = [False] * len(panel_masks)
        for i in range(len(fake_imgs)):

            x, y, w, h = masks[i]
            # print(x,y,w,h)

            # fake_img = io.imread(fake_imgs[i])
            # fake_img = np.array(fake_img)
            fake_img = fake_imgs[i]
            panel_mask = panel_masks[i]
            
            try:
                img[y:y + h, x:x + w][panel_mask == 0] = fake_img[panel_mask == 0]
                mask_used[i] = True
            except:
                print(f"Shape mismatch: fake_img shape: {fake_img.shape}, panel_mask shape: {panel_mask.shape}, img shape: {img.shape}")
                # Attempt to find a matching subarray
                for j in range(len(panel_masks)):

                    if not mask_used[j]:

                          sub_panel_mask = panel_masks[j]
                          print(sub_panel_mask.shape, fake_img.shape[:2])
                          if fake_img.shape[:2] == sub_panel_mask.shape:
                              x, y, w, h = masks[j]
                        
                              img[y:y + h, x:x + w][sub_panel_mask == 0] = fake_img[sub_panel_mask == 0]
                              mask_used[j] = True
                              break
                    
        print(mask_used)           
                # if not matched:
                #     print("No matching subarray found for fake_img and panel_mask")

            # Image.fromarray(img).show()
        out_folder = os.path.dirname(img_file)
        out_name = os.path.basename(img_file)
        out_name = os.path.splitext(out_name)[0]
        out_img_path = os.path.join(out_folder,'color',f'{out_name}_color.png')

        # show image
        # Image.fromarray(img).show()
        # save image
        folder_path = os.path.join(out_folder, 'color')
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        io.imsave(out_img_path, img)

