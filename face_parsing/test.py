#!/usr/bin/python
# -*- encoding: utf-8 -*-
import numpy as np
from model import BiSeNet

import torch

import os
import os.path as osp

from PIL import Image
import torchvision.transforms as transforms
import cv2
from pathlib import Path
import configargparse
import tqdm

# import ttach as tta


import numpy as np
from scipy import ndimage

# Example mask and label indices for demonstration

def get_top_20_percent_of_neck(mask):
    neck_labels = [14]

    # Isolate the neck
    neck_mask = np.isin(mask, neck_labels)

    # Identify unique x-coordinates (columns) where the neck is present
    unique_x_coords = np.unique(np.where(neck_mask)[1])

    # Initialize an empty mask for the top 10% of the neck across all x-coordinates
    top_10_percent_neck_mask = np.zeros_like(mask, dtype=bool)

    # Iterate over each unique x-coordinate
    for x in unique_x_coords:
        # Find y-coordinates (rows) of neck pixels at this x-coordinate
        y_coords = np.where(neck_mask[:, x])[0]
        
        # Calculate the number of pixels that make up the top 10%
        top_10_percent_count = int(np.ceil(0.4 * len(y_coords)))
        
        # If there are neck pixels at this x-coordinate
        if top_10_percent_count > 0:
            # Sort y-coordinates in ascending order (top of the image has lower y-values)
            sorted_y_coords = np.sort(y_coords)
            
            # Select the top 10% based on y-coordinates
            top_y_coords = sorted_y_coords[:top_10_percent_count]
            
            # Mark these pixels in the top 10% mask
            top_10_percent_neck_mask[top_y_coords, x] = True
    # Label connected components
    label_im, nb_labels = ndimage.label(top_10_percent_neck_mask)

    # Find the size of each component
    sizes = ndimage.sum(top_10_percent_neck_mask, label_im, range(nb_labels + 1))

    # Exclude the background label by setting its size to 0
    sizes[0] = 0

    # Find the label of the largest component
    largest_component_label = np.argmax(sizes)

    # Create a mask of the largest component
    largest_region_mask = (label_im == largest_component_label)
    return largest_region_mask


def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg', img_size=(512, 512)):
    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(
        vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros(
        (vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + np.array([255, 255, 255])  # + 255
    vis_parsing_anno_color_onlyface = vis_parsing_anno_color.copy()
    torso_vis_parsing_anno_color_onlyface = np.zeros(
        (vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) 
    num_of_class = np.max(vis_parsing_anno)
    # ['bg', 'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
    # 'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
    # print(num_of_class)
    
    for pi in range(1, 14):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = np.array([255, 0, 0])
        torso_vis_parsing_anno_color_onlyface[index[0], index[1], :] = np.array([255, 255, 255])
    # only face
    # vis_parsing_anno_color_onlyface = vis_parsing_anno_color.copy()
    
    for pi in range(1, 7):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color_onlyface[index[0], index[1], :] = np.array([255, 0, 0])
    for pi in range(10, 14):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color_onlyface[index[0], index[1], :] = np.array([255, 0, 0])
    
    top_20_percent_of_neck = get_top_20_percent_of_neck(vis_parsing_anno)
    index = np.where(top_20_percent_of_neck == True)
    vis_parsing_anno_color_onlyface[index[0], index[1], :] = np.array([255, 0, 0])
    
    for pi in range(14, 16):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = np.array([0, 255, 0])
    for pi in range(16, 17):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = np.array([0, 0, 255])
    for pi in range(17, num_of_class+1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = np.array([255, 0, 0])
        torso_vis_parsing_anno_color_onlyface[index[0], index[1], :] = np.array([255, 255, 255])

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_parsing_anno_color_onlyface = vis_parsing_anno_color_onlyface.astype(np.uint8)
    torso_vis_parsing_anno_color_onlyface = torso_vis_parsing_anno_color_onlyface.astype(np.uint8)
    index = np.where(vis_parsing_anno == num_of_class-1)
    vis_im = cv2.resize(vis_parsing_anno_color, img_size)
    torso_vis_parsing_anno_color_onlyface = cv2.resize(torso_vis_parsing_anno_color_onlyface, img_size)
    vis_im_onlyface = cv2.resize(vis_parsing_anno_color_onlyface, img_size)
    if save_im:
        save_face_mask_torso = save_path.replace('parsing', 'face_mask')
        print("save_face_mask_torso: ", save_face_mask_torso)
        cv2.imwrite(save_face_mask_torso, torso_vis_parsing_anno_color_onlyface)
        save_path_face = save_path.replace('.png', '_face.png')
        cv2.imwrite(save_path, vis_im)
        blurred_face = cv2.GaussianBlur(vis_im_onlyface, (99, 99), 2)
        cv2.imwrite(save_path_face, blurred_face)


def evaluate(respth='./res/test_res', dspth='./data', cp='model_final_diss.pth'):

    Path(respth).mkdir(parents=True, exist_ok=True)
    print(f'[INFO] output path: {respth} from {dspth}')
    print(f'[INFO] loading model...')
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net.load_state_dict(torch.load(cp))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    image_paths = os.listdir(dspth)
    print("image_paths: ", image_paths)
    with torch.no_grad():
        for image_path in tqdm.tqdm(image_paths):
            if image_path.endswith('.jpg') or image_path.endswith('.png'):
                img = Image.open(osp.join(dspth, image_path))
                ori_size = img.size
                image = img.resize((512, 512), Image.BILINEAR)
                image = image.convert("RGB")
                img = to_tensor(image)

                # test-time augmentation.
                inputs = torch.unsqueeze(img, 0) # [1, 3, 512, 512]
                outputs = net(inputs.cuda())
                parsing = outputs.mean(0).cpu().numpy().argmax(0)

                image_path = int(image_path[:-4])
                image_path = str(image_path) + '.png'

                vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(respth, image_path), img_size=ori_size)


if __name__ == "__main__":
    parser = configargparse.ArgumentParser()
    parser.add_argument('--respath', type=str, default='./result/', help='result path for label')
    parser.add_argument('--imgpath', type=str, default='./imgs/', help='path for input images')
    parser.add_argument('--modelpath', type=str, default='data_utils/face_parsing/79999_iter.pth')
    parser.add_argument('--resolution', type=int, default=512, help='resolution of input image')
    args = parser.parse_args()
    evaluate(respth=args.respath, dspth=args.imgpath, cp=args.modelpath)
