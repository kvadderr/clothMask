from network import U2NET

import os
from PIL import Image
import cv2
import gdown
import argparse
import numpy as np
import io
import base64

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from collections import OrderedDict
from options import opt


def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return
    model_state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    return model


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


class Normalize_image(object):
    """Normalize given tensor into given mean and standard dev

    Args:
        mean (float): Desired mean to substract from tensors
        std (float): Desired std to divide from tensors
    """

    def __init__(self, mean, std):
        assert isinstance(mean, (float))
        if isinstance(mean, float):
            self.mean = mean

        if isinstance(std, float):
            self.std = std

        self.normalize_1 = transforms.Normalize(self.mean, self.std)
        self.normalize_3 = transforms.Normalize([self.mean] * 3, [self.std] * 3)
        self.normalize_18 = transforms.Normalize([self.mean] * 18, [self.std] * 18)

    def __call__(self, image_tensor):
        if image_tensor.shape[0] == 1:
            return self.normalize_1(image_tensor)

        elif image_tensor.shape[0] == 3:
            return self.normalize_3(image_tensor)

        elif image_tensor.shape[0] == 18:
            return self.normalize_18(image_tensor)

        else:
            assert "Please set proper channels! Normlization implemented only for 1, 3 and 18"




def apply_transform(img):
    transforms_list = []
    transforms_list += [transforms.ToTensor()]
    transforms_list += [Normalize_image(0.5, 0.5)]
    transform_rgb = transforms.Compose(transforms_list)
    return transform_rgb(img)



def generate_mask(input_image, net, palette, device = 'cpu'):

    #img = Image.open(input_image).convert('RGB')
    img = input_image
    img_size = img.size
    img = img.resize((768, 768), Image.BICUBIC)
    image_tensor = apply_transform(img)
    image_tensor = torch.unsqueeze(image_tensor, 0)

    alpha_out_dir = os.path.join(opt.output,'alpha')
    cloth_seg_out_dir = os.path.join(opt.output,'cloth_seg')

    os.makedirs(alpha_out_dir, exist_ok=True)
    os.makedirs(cloth_seg_out_dir, exist_ok=True)

    with torch.no_grad():
        output_tensor = net(image_tensor.to(device))
        output_tensor = F.log_softmax(output_tensor[0], dim=1)
        output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
        output_tensor = torch.squeeze(output_tensor, dim=0)
        output_arr = output_tensor.cpu().numpy()

    kernel = np.ones((30, 30), np.uint8)
    dilated_mask = cv2.dilate(output_arr[0].astype(np.uint8), kernel, iterations=1)
    output_arr[0] = dilated_mask
    classes_to_save = []

    # Check which classes are present in the image
    for cls in range(1, 4):  # Exclude background class (0)
        if np.any(output_arr == cls):
            classes_to_save.append(cls)

    # Save alpha masks
    for cls in classes_to_save:
        alpha_mask = (output_arr == cls).astype(np.uint8) * 255
        alpha_mask = alpha_mask[0]  # Selecting the first channel to make it 2D
        alpha_mask_img = Image.fromarray(alpha_mask, mode='L')
        alpha_mask_img = alpha_mask_img.resize(img_size, Image.BICUBIC)
        alpha_mask_img.save(os.path.join(alpha_out_dir, f'{cls}.png'))

    # Save final cloth segmentations
    cloth_seg = Image.fromarray(output_arr[0].astype(np.uint8), mode='P')
    cloth_seg.putpalette([0, 0, 0, 255, 255, 255])
    cloth_seg = cloth_seg.resize(img_size, Image.BICUBIC)
    cloth_seg.save(os.path.join(cloth_seg_out_dir, 'final_seg.png'))
    return cloth_seg

def generate_mask_base64(input_image, net, palette, device='cpu'):
    img = input_image
    img_size = img.size
    img = img.resize((768, 768), Image.BICUBIC)
    image_tensor = apply_transform(img)
    image_tensor = torch.unsqueeze(image_tensor, 0)

    with torch.no_grad():
        output_tensor = net(image_tensor.to(device))
        output_tensor = F.log_softmax(output_tensor[0], dim=1)
        output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
        output_tensor = torch.squeeze(output_tensor, dim=0)
        output_arr = output_tensor.cpu().numpy()

    kernel = np.ones((30, 30), np.uint8)
    dilated_mask = cv2.dilate(output_arr[0].astype(np.uint8), kernel, iterations=1)
    output_arr[0] = dilated_mask

    # Convert cloth segmentation image to PIL Image
    cloth_seg_pil = Image.fromarray(output_arr[0].astype(np.uint8), mode='P')
    cloth_seg_pil.putpalette([0, 0, 0, 255, 255, 255])
    cloth_seg_pil = cloth_seg_pil.resize(img_size, Image.BICUBIC)

    # Convert PIL Image to bytes
    buffer = io.BytesIO()
    cloth_seg_pil.save(buffer, format="PNG")
    cloth_seg_bytes = buffer.getvalue()

    # Encode the bytes to base64
    cloth_seg_base64 = base64.b64encode(cloth_seg_bytes).decode("utf-8")

    return cloth_seg_base64

def check_or_download_model(file_path):
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        url = "https://drive.google.com/uc?id=11xTBALOeUkyuaK3l60CpkYHLTmv7k3dY"
        gdown.download(url, file_path, quiet=False)


def load_seg_model(checkpoint_path, device='cpu'):
    net = U2NET(in_ch=3, out_ch=4)
    check_or_download_model(checkpoint_path)
    net = load_checkpoint(net, checkpoint_path)
    net = net.to(device)
    net = net.eval()

    return net


def main(args):

    device = 'cuda:0' if args.cuda else 'cpu'

    # Create an instance of your model
    model = load_seg_model(args.checkpoint_path, device=device)

    palette = get_palette(4)

    img = Image.open(args.image).convert('RGB')

    cloth_seg_base64 = generate_mask_base64(img, net=model, palette=palette, device=device)
    print(cloth_seg_base64)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Help to set arguments for Cloth Segmentation.')
    parser.add_argument('--image', type=str, help='Path to the input image')
    parser.add_argument('--cuda', action='store_true', help='Enable CUDA (default: False)')
    parser.add_argument('--checkpoint_path', type=str, default='model/cloth_segm.pth', help='Path to the checkpoint file')
    args = parser.parse_args()

    main(args)