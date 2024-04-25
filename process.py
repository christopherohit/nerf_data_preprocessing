import os
import glob
import tqdm
import json
import argparse
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import subprocess
import traceback



model_path = 'face_landmarker_v2_with_blendshapes.task'
if not os.path.exists(model_path):
    cmd = f"wget -O {model_path} -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    subprocess.call(cmd, shell=True)

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a face landmarker instance with the video mode:
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO)


landmark_points_68 = [162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,71,63,105,66,107,336,
                  296,334,293,301,168,197,5,4,75,97,2,326,305,33,160,158,133,153,144,362,385,387,263,373,
                  380,61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87]



def get_landmark(full_frame, frame_idx):
    landmarks_extracted = []
    try:
        with FaceLandmarker.create_from_options(options) as landmarker:
            h, w = full_frame.shape[0], full_frame.shape[1]
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=full_frame.astype(np.uint8))
            frame_timestamp_ms = int(1000 / 25 ) * frame_idx
            face_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            if not face_landmarker_result.face_landmarks:
                raise Exception("No face  in frame 0")  # not detect face
            face_landmarks = face_landmarker_result.face_landmarks[0]
            
            for idx, landmark in enumerate(face_landmarks):
                
                
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                z = landmark.z
                if idx in landmark_points_68:
                    landmarks_extracted.append([idx, x, y])
                
            landmarks_extracted = sorted(landmarks_extracted, key=lambda x: landmark_points_68.index(x[0]))
            # remove index
            landmarks_extracted = [x[1:] for x in landmarks_extracted]
            
    except Exception as e:
        print("Error in frame", frame_idx, e)
        traceback.print_exc()
    return np.array(landmarks_extracted)



def extract_audio(path, out_path, sample_rate=16000):
    
    print(f'[INFO] ===== extract audio from {path} to {out_path} =====')
    cmd = f'ffmpeg -i {path} -f wav -ar {sample_rate} {out_path}'
    os.system(cmd)
    print(f'[INFO] ===== extracted audio =====')


def extract_audio_features(path):
    cmd = f'python data_utils/extract_audio_visual.py --audio_path {path}'
    os.system(cmd)
    print(f'[INFO] ===== extracted audio labels =====')

def extract_images(path, out_path, fps=25):

    print(f'[INFO] ===== extract images from {path} to {out_path} =====')
    cmd = f'ffmpeg -i {path} -vf fps={fps} -qmin 1 -q:v 1 -start_number 0 {os.path.join(out_path, "%d.jpg")}'
    os.system(cmd)
    print(f'[INFO] ===== extracted images =====')


def extract_semantics(ori_imgs_dir, parsing_dir, resolution):

    print(f'[INFO] ===== extract semantics from {ori_imgs_dir} to {parsing_dir} =====')
    cmd = f'python data_utils/face_parsing/test.py --respath={parsing_dir} --imgpath={ori_imgs_dir} --resolution={resolution}'
    os.system(cmd)
    print(f'[INFO] ===== extracted semantics =====')


def extract_landmarks(ori_imgs_dir, is_use_mediapipe=False):

    print(f'[INFO] ===== extract face landmarks from {ori_imgs_dir} =====')
    if not is_use_mediapipe:
        import face_alignment
        try:
            fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
        except:
            fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
        image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))
        for image_path in tqdm.tqdm(image_paths):
            input = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) # [H, W, 3]
            input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
            preds = fa.get_landmarks(input)
            if len(preds) > 0:
                lands = preds[0].reshape(-1, 2)[:,:2]
                np.savetxt(image_path.replace('jpg', 'lms'), lands, '%f')
        del fa
        print(f'[INFO] ===== extracted face landmarks =====')
    else:
        image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))
        image_paths = sorted(image_paths, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for image_path in tqdm.tqdm(image_paths):
            full_frame = cv2.imread(image_path)
            frame_idx = int(image_path.split('/')[-1].split('.')[0])
            landmark = get_landmark(full_frame, frame_idx)
            np.savetxt(image_path.replace('jpg', 'lms'), landmark, '%f')


def extract_blendshape(base_dir, ori_imgs_dir):
    
    model_bsList = ["browDownLeft",
                "browDownRight",
                "browInnerUp",
                "browOuterUpLeft",
                "browOuterUpRight",
                "cheekPuff",
                "cheekSquintLeft",
                "cheekSquintRight",
                "eyeBlinkLeft",
                "eyeBlinkRight",
                "eyeLookDownLeft",
                "eyeLookDownRight",
                "eyeLookInLeft",
                "eyeLookInRight",
                "eyeLookOutLeft",
                "eyeLookOutRight",
                "eyeLookUpLeft",
                "eyeLookUpRight",
                "eyeSquintLeft",
                "eyeSquintRight",
                "eyeWideLeft",
                "eyeWideRight",
                "jawForward",
                "jawLeft",
                "jawOpen",
                "jawRight",
                "mouthClose",
                "mouthDimpleLeft",
                "mouthDimpleRight",
                "mouthFrownLeft",
                "mouthFrownRight",
                "mouthFunnel",
                "mouthLeft",
                "mouthLowerDownLeft",
                "mouthLowerDownRight",
                "mouthPressLeft",
                "mouthPressRight",
                "mouthPucker",
                "mouthRight",
                "mouthRollLower",
                "mouthRollUpper",
                "mouthShrugLower",
                "mouthShrugUpper",
                "mouthSmileLeft",
                "mouthSmileRight",
                "mouthStretchLeft",
                "mouthStretchRight",
                "mouthUpperUpLeft",
                "mouthUpperUpRight",
                "noseSneerLeft",
                "noseSneerRight",
                "tongueOut"]

    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))
    # STEP 1: Import the necessary modules.
    

    # STEP 2: Create an FaceLandmarker object.
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                        output_face_blendshapes=True,
                                        output_facial_transformation_matrixes=True,
                                        num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)
    list_bs = np.zeros((len(image_paths), 52))
    # STEP 3: Load the input image.
    for img_path in image_paths:
        index_img = int(img_path.split('/')[-1].split('.')[0])
        
        image = mp.Image.create_from_file(img_path)
        detection_result = detector.detect(image)
        blend_result = detection_result.face_blendshapes[0]
        bs = np.zeros(52)
        for i in range(len(blend_result)):
            if blend_result[i].category_name in model_bsList:
                bs[model_bsList.index(blend_result[i].category_name)] = blend_result[i].score
            else:
                bs[model_bsList.index("tongueOut")] = blend_result[i].score
        list_bs[index_img] = bs
    np.save(base_dir + '/bs.npy', list_bs)
        
    


def extract_background(base_dir, ori_imgs_dir):
    
    print(f'[INFO] ===== extract background image from {ori_imgs_dir} =====')

    from sklearn.neighbors import NearestNeighbors

    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))
    # only use 1/20 image_paths 
    image_paths = image_paths[::20]
    # read one image to get H/W
    tmp_image = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED) # [H, W, 3]
    h, w = tmp_image.shape[:2]

    # nearest neighbors
    all_xys = np.mgrid[0:h, 0:w].reshape(2, -1).transpose()
    distss = []
    for image_path in tqdm.tqdm(image_paths):
        parse_img = cv2.imread(image_path.replace('ori_imgs', 'parsing').replace('.jpg', '.png'))
        bg = (parse_img[..., 0] == 255) & (parse_img[..., 1] == 255) & (parse_img[..., 2] == 255)
        fg_xys = np.stack(np.nonzero(~bg)).transpose(1, 0)
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(fg_xys)
        dists, _ = nbrs.kneighbors(all_xys)
        distss.append(dists)

    distss = np.stack(distss)
    max_dist = np.max(distss, 0)
    max_id = np.argmax(distss, 0)

    bc_pixs = max_dist > 5
    bc_pixs_id = np.nonzero(bc_pixs)
    bc_ids = max_id[bc_pixs]

    imgs = []
    num_pixs = distss.shape[1]
    for image_path in image_paths:
        img = cv2.imread(image_path)
        imgs.append(img)
    imgs = np.stack(imgs).reshape(-1, num_pixs, 3)

    bc_img = np.zeros((h*w, 3), dtype=np.uint8)
    bc_img[bc_pixs_id, :] = imgs[bc_ids, bc_pixs_id, :]
    bc_img = bc_img.reshape(h, w, 3)

    max_dist = max_dist.reshape(h, w)
    bc_pixs = max_dist > 5
    bg_xys = np.stack(np.nonzero(~bc_pixs)).transpose()
    fg_xys = np.stack(np.nonzero(bc_pixs)).transpose()
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(fg_xys)
    distances, indices = nbrs.kneighbors(bg_xys)
    bg_fg_xys = fg_xys[indices[:, 0]]
    bc_img[bg_xys[:, 0], bg_xys[:, 1], :] = bc_img[bg_fg_xys[:, 0], bg_fg_xys[:, 1], :]

    cv2.imwrite(os.path.join(base_dir, 'bc.jpg'), bc_img)

    print(f'[INFO] ===== extracted background image =====')


def extract_torso_and_gt(base_dir, ori_imgs_dir):

    print(f'[INFO] ===== extract torso and gt images for {base_dir} =====')

    from scipy.ndimage import binary_erosion, binary_dilation

    # load bg
    bg_image = cv2.imread(os.path.join(base_dir, 'bc.jpg'), cv2.IMREAD_UNCHANGED)
    
    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))

    for image_path in tqdm.tqdm(image_paths):
        # read ori image
        ori_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) # [H, W, 3]

        # read semantics
        seg = cv2.imread(image_path.replace('ori_imgs', 'parsing').replace('.jpg', '.png'))
        head_part = (seg[..., 0] == 255) & (seg[..., 1] == 0) & (seg[..., 2] == 0)
        neck_part = (seg[..., 0] == 0) & (seg[..., 1] == 255) & (seg[..., 2] == 0)
        torso_part = (seg[..., 0] == 0) & (seg[..., 1] == 0) & (seg[..., 2] == 255)
        bg_part = (seg[..., 0] == 255) & (seg[..., 1] == 255) & (seg[..., 2] == 255)

        # get gt image
        gt_image = ori_image.copy()
        gt_image[bg_part] = bg_image[bg_part]
        cv2.imwrite(image_path.replace('ori_imgs', 'gt_imgs'), gt_image)

        # get torso image
        torso_image = gt_image.copy() # rgb
        torso_image[head_part] = bg_image[head_part]
        torso_alpha = 255 * np.ones((gt_image.shape[0], gt_image.shape[1], 1), dtype=np.uint8) # alpha
        
        # torso part "vertical" in-painting...
        L = 8 + 1
        torso_coords = np.stack(np.nonzero(torso_part), axis=-1) # [M, 2]
        # lexsort: sort 2D coords first by y then by x, 
        # ref: https://stackoverflow.com/questions/2706605/sorting-a-2d-numpy-array-by-multiple-axes
        inds = np.lexsort((torso_coords[:, 0], torso_coords[:, 1]))
        torso_coords = torso_coords[inds]
        # choose the top pixel for each column
        u, uid, ucnt = np.unique(torso_coords[:, 1], return_index=True, return_counts=True)
        top_torso_coords = torso_coords[uid] # [m, 2]
        # only keep top-is-head pixels
        top_torso_coords_up = top_torso_coords.copy() - np.array([1, 0])
        mask = head_part[tuple(top_torso_coords_up.T)] 
        if mask.any():
            top_torso_coords = top_torso_coords[mask]
            # get the color
            top_torso_colors = gt_image[tuple(top_torso_coords.T)] # [m, 3]
            # construct inpaint coords (vertically up, or minus in x)
            inpaint_torso_coords = top_torso_coords[None].repeat(L, 0) # [L, m, 2]
            inpaint_offsets = np.stack([-np.arange(L), np.zeros(L, dtype=np.int32)], axis=-1)[:, None] # [L, 1, 2]
            inpaint_torso_coords += inpaint_offsets
            inpaint_torso_coords = inpaint_torso_coords.reshape(-1, 2) # [Lm, 2]
            inpaint_torso_colors = top_torso_colors[None].repeat(L, 0) # [L, m, 3]
            darken_scaler = 0.98 ** np.arange(L).reshape(L, 1, 1) # [L, 1, 1]
            inpaint_torso_colors = (inpaint_torso_colors * darken_scaler).reshape(-1, 3) # [Lm, 3]
            # set color
            torso_image[tuple(inpaint_torso_coords.T)] = inpaint_torso_colors

            inpaint_torso_mask = np.zeros_like(torso_image[..., 0]).astype(bool)
            inpaint_torso_mask[tuple(inpaint_torso_coords.T)] = True
        else:
            inpaint_torso_mask = None
            

        # neck part "vertical" in-painting...
        push_down = 4
        L = 48 + push_down + 1

        neck_part = binary_dilation(neck_part, structure=np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool), iterations=3)

        neck_coords = np.stack(np.nonzero(neck_part), axis=-1) # [M, 2]
        # lexsort: sort 2D coords first by y then by x, 
        # ref: https://stackoverflow.com/questions/2706605/sorting-a-2d-numpy-array-by-multiple-axes
        inds = np.lexsort((neck_coords[:, 0], neck_coords[:, 1]))
        neck_coords = neck_coords[inds]
        # choose the top pixel for each column
        u, uid, ucnt = np.unique(neck_coords[:, 1], return_index=True, return_counts=True)
        top_neck_coords = neck_coords[uid] # [m, 2]
        # only keep top-is-head pixels
        top_neck_coords_up = top_neck_coords.copy() - np.array([1, 0])
        mask = head_part[tuple(top_neck_coords_up.T)] 
        
        top_neck_coords = top_neck_coords[mask]
        # push these top down for 4 pixels to make the neck inpainting more natural...
        offset_down = np.minimum(ucnt[mask] - 1, push_down)
        top_neck_coords += np.stack([offset_down, np.zeros_like(offset_down)], axis=-1)
        # get the color
        top_neck_colors = gt_image[tuple(top_neck_coords.T)] # [m, 3]
        # construct inpaint coords (vertically up, or minus in x)
        inpaint_neck_coords = top_neck_coords[None].repeat(L, 0) # [L, m, 2]
        inpaint_offsets = np.stack([-np.arange(L), np.zeros(L, dtype=np.int32)], axis=-1)[:, None] # [L, 1, 2]
        inpaint_neck_coords += inpaint_offsets
        inpaint_neck_coords = inpaint_neck_coords.reshape(-1, 2) # [Lm, 2]
        """
        neck_avg_color = np.mean(gt_image[neck_part], axis=0)
        inpaint_neck_colors = top_neck_colors[None].repeat(L, 0)
        alpha_values = np.linspace(1, 0, L).reshape(L, 1, 1) 
        inpaint_neck_colors = inpaint_neck_colors * alpha_values + neck_avg_color * (1 - alpha_values)
        inpaint_neck_colors = inpaint_neck_colors.reshape(-1, 3)
        torso_image[tuple(inpaint_neck_coords.T)] = inpaint_neck_colors
        """
        neck_avg_color = np.mean(gt_image[neck_part], axis=0)
        inpaint_neck_colors = top_neck_colors[None].repeat(L, 0) # [L, m, 3]
        alpha_values = np.linspace(1, 0, L).reshape(L, 1, 1) 
        # darken_scaler = 0.98 ** np.arange(L).reshape(L, 1, 1) # [L, 1, 1]
        # inpaint_neck_colors = (inpaint_neck_colors * darken_scaler).reshape(-1, 3) # [Lm, 3]
        inpaint_neck_colors = inpaint_neck_colors * alpha_values + neck_avg_color * (1 - alpha_values)
        inpaint_neck_colors = inpaint_neck_colors.reshape(-1, 3)
        # set color
        torso_image[tuple(inpaint_neck_coords.T)] = inpaint_neck_colors

        # apply blurring to the inpaint area to avoid vertical-line artifects...
        inpaint_mask = np.zeros_like(torso_image[..., 0]).astype(bool)
        inpaint_mask[tuple(inpaint_neck_coords.T)] = True

        blur_img = torso_image.copy()
        blur_img = cv2.GaussianBlur(blur_img, (5, 5), cv2.BORDER_DEFAULT)

        torso_image[inpaint_mask] = blur_img[inpaint_mask]

        # set mask
        mask = (neck_part | torso_part | inpaint_mask)
        if inpaint_torso_mask is not None:
            mask = mask | inpaint_torso_mask
        torso_image[~mask] = 0
        torso_alpha[~mask] = 0

        cv2.imwrite(image_path.replace('ori_imgs', 'torso_imgs').replace('.jpg', '.png'), np.concatenate([torso_image, torso_alpha], axis=-1))

    print(f'[INFO] ===== extracted torso and gt images =====')


def face_tracking(ori_imgs_dir):

    print(f'[INFO] ===== perform face tracking =====')

    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))
    
    # read one image to get H/W
    tmp_image = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED) # [H, W, 3]
    h, w = tmp_image.shape[:2]

    cmd = f'python data_utils/face_tracking/face_tracker.py --path={ori_imgs_dir} --img_h={h} --img_w={w} --frame_num={len(image_paths)}'

    os.system(cmd)

    print(f'[INFO] ===== finished face tracking =====')


def save_transforms(base_dir, ori_imgs_dir):
    print(f'[INFO] ===== save transforms =====')

    import torch

    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))
    
    # read one image to get H/W
    tmp_image = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED) # [H, W, 3]
    h, w = tmp_image.shape[:2]

    params_dict = torch.load(os.path.join(base_dir, 'track_params.pt'))
    focal_len = params_dict['focal']
    euler_angle = params_dict['euler']
    trans = params_dict['trans'] / 10.0
    valid_num = euler_angle.shape[0]

    def euler2rot(euler_angle):
        batch_size = euler_angle.shape[0]
        theta = euler_angle[:, 0].reshape(-1, 1, 1)
        phi = euler_angle[:, 1].reshape(-1, 1, 1)
        psi = euler_angle[:, 2].reshape(-1, 1, 1)
        one = torch.ones((batch_size, 1, 1), dtype=torch.float32, device=euler_angle.device)
        zero = torch.zeros((batch_size, 1, 1), dtype=torch.float32, device=euler_angle.device)
        rot_x = torch.cat((
            torch.cat((one, zero, zero), 1),
            torch.cat((zero, theta.cos(), theta.sin()), 1),
            torch.cat((zero, -theta.sin(), theta.cos()), 1),
        ), 2)
        rot_y = torch.cat((
            torch.cat((phi.cos(), zero, -phi.sin()), 1),
            torch.cat((zero, one, zero), 1),
            torch.cat((phi.sin(), zero, phi.cos()), 1),
        ), 2)
        rot_z = torch.cat((
            torch.cat((psi.cos(), -psi.sin(), zero), 1),
            torch.cat((psi.sin(), psi.cos(), zero), 1),
            torch.cat((zero, zero, one), 1)
        ), 2)
        return torch.bmm(rot_x, torch.bmm(rot_y, rot_z))


    # train_val_split = int(valid_num*0.5)
    # train_val_split = valid_num - 25 * 20 # take the last 20s as valid set.
    train_val_split = int(valid_num * 10 / 11)

    train_ids = torch.arange(0, train_val_split)
    val_ids = torch.arange(train_val_split, valid_num)

    rot = euler2rot(euler_angle)
    rot_inv = rot.permute(0, 2, 1)
    trans_inv = -torch.bmm(rot_inv, trans.unsqueeze(2))

    pose = torch.eye(4, dtype=torch.float32)
    save_ids = ['train', 'val']
    train_val_ids = [train_ids, val_ids]
    mean_z = -float(torch.mean(trans[:, 2]).item())

    for split in range(2):
        transform_dict = dict()
        transform_dict['focal_len'] = float(focal_len[0])
        transform_dict['cx'] = float(w/2.0)
        transform_dict['cy'] = float(h/2.0)
        transform_dict['frames'] = []
        ids = train_val_ids[split]
        save_id = save_ids[split]

        for i in ids:
            i = i.item()
            frame_dict = dict()
            frame_dict['img_id'] = i
            frame_dict['aud_id'] = i

            pose[:3, :3] = rot_inv[i]
            pose[:3, 3] = trans_inv[i, :, 0]

            frame_dict['transform_matrix'] = pose.numpy().tolist()

            transform_dict['frames'].append(frame_dict)

        with open(os.path.join(base_dir, 'transforms_' + save_id + '.json'), 'w') as fp:
            json.dump(transform_dict, fp, indent=2, separators=(',', ': '))

    print(f'[INFO] ===== finished saving transforms =====')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="path to video file")
    parser.add_argument('--task', type=int, default=-1, help="-1 means all")
    parser.add_argument('--resolution', type=int, default=512, help="resolution of the images")
    parser.add_argument('--is_use_mediapipe', action='store_true', help="mediapie")

    opt = parser.parse_args()

    base_dir = os.path.dirname(opt.path)
    resolution = opt.resolution
    wav_path = os.path.join(base_dir, 'aud.wav')
    ori_imgs_dir = os.path.join(base_dir, 'ori_imgs')
    parsing_dir = os.path.join(base_dir, 'parsing')
    gt_imgs_dir = os.path.join(base_dir, 'gt_imgs')
    torso_imgs_dir = os.path.join(base_dir, 'torso_imgs')
    face_mask_dir = os.path.join(base_dir, 'face_mask')

    os.makedirs(ori_imgs_dir, exist_ok=True)
    os.makedirs(parsing_dir, exist_ok=True)
    os.makedirs(gt_imgs_dir, exist_ok=True)
    os.makedirs(torso_imgs_dir, exist_ok=True)
    os.makedirs(face_mask_dir, exist_ok=True)


    # extract audio
    if opt.task == -1 or opt.task == 1:
        extract_audio(opt.path, wav_path)

    # extract audio features
    if opt.task == -1 or opt.task == 2:
        extract_audio_features(wav_path)

    # extract images
    if opt.task == -1 or opt.task == 3:
        extract_images(opt.path, ori_imgs_dir)

    # face parsing
    if opt.task == -1 or opt.task == 4:
        extract_semantics(ori_imgs_dir, parsing_dir, resolution)

    # extract bg
    if opt.task == -1 or opt.task == 5:
        extract_background(base_dir, ori_imgs_dir)

    if opt.task == -1 or opt.task == 10:
        extract_blendshape(base_dir, ori_imgs_dir)
        
    # extract torso images and gt_images
    if opt.task == -1 or opt.task == 6:
        extract_torso_and_gt(base_dir, ori_imgs_dir)

    # extract face landmarks
    if opt.task == -1 or opt.task == 7:
        extract_landmarks(ori_imgs_dir, is_use_mediapipe=opt.is_use_mediapipe)

    # face tracking
    if opt.task == -1 or opt.task == 8:
        face_tracking(ori_imgs_dir)

    # save transforms.json
    if opt.task == -1 or opt.task == 9:
        save_transforms(base_dir, ori_imgs_dir)

