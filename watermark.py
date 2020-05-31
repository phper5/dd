import torch
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curPath+"/vmtest")

import os.path
from vmtest.utils.train_utils import load_globals, init_folders, init_nets
from vmtest.loaders.motif_dataset import MotifDS
from PIL import Image
import numpy as np
import uuid
import shutil
device = torch.device('cuda:0')
device = torch.device('cpu')


def load_image(image_path, _device, include_tensor=False):
    numpy_image = None
    tensor_image = None
    if os.path.isfile(image_path):
        to_save = False
        row_image = Image.open(image_path)
        w, h = row_image.size
        if h > 512:
            to_save = True
            h = int((512. * h) / w)
            row_image = row_image.resize((512, h), Image.BICUBIC)
        w, h = row_image.size
        if w % 16 != 0 or h % 16 != 0:
            to_save = True
            row_image = row_image.crop((0, 0, (w // 16) * 16, (h // 16) * 16))
        if to_save:
            row_image.save(image_path)
        numpy_image = np.array(row_image)
        if len(numpy_image.shape) != 3:
            numpy_image = np.repeat(np.expand_dims(numpy_image, 2), 3, axis=2)
        if numpy_image.shape[2] != 3:
            numpy_image = numpy_image[:, :, :3]
        if include_tensor:
            tensor_image = MotifDS.trans(MotifDS.flip(numpy_image)[0])[0]
            tensor_image = torch.unsqueeze(torch.from_numpy(tensor_image), 0).to(_device)
        numpy_image = np.expand_dims(numpy_image / 255, 0)
    return numpy_image, tensor_image


def transform_to_numpy_image(tensor_image):
    image = tensor_image.cpu().detach().numpy()
    image = np.transpose(image, (0, 2, 3, 1))
    if image.shape[3] != 3:
        image = np.repeat(image, 3, axis=3)
    else:
        image = (image / 2 + 0.5)
    return image


def collect_synthesized(_source):
    paths = []
    for root, _, files in os.walk(_source):
            for file in files:
                file_name, file_extension = os.path.splitext(file)
                if (file_extension == '.png' or file_extension == '.jpg' or file_extension == '.jpeg') and \
                   ('real' not in file_name and 'reconstructed' not in file_name and 'grid' not in file_name):
                    paths.append(os.path.join(root, file))
    return paths

def save_numpy_image(image, image_suffix,target_file_path):
    image = (image * 255).astype(np.uint8)  # unnormalize
    path,filename = os.path.split(target_file_path)
    _,ext = os.path.splitext(filename)
    image_path = '%s/%s.png' % (path, image_suffix+str(uuid.uuid4()) )
    image = Image.fromarray(image)
    image.save(image_path)
    return image_path

def vmtest(file_path,net_path):
    opt = load_globals(net_path, {}, override=False)
    net = init_nets(opt, net_path, device, '').eval()
    sy_np,sy_ts = load_image(file_path, device, True)
    results = list(net(sy_ts))
    for idx, result in enumerate(results):
        results[idx] = transform_to_numpy_image(result)
    reconstructed_mask = results[1]
    reconstructed_motif = None
    if len(results) == 3:
        reconstructed_raw_motif = results[2]
        reconstructed_motif = (reconstructed_raw_motif - 1) * reconstructed_mask + 1
    reconstructed_image = reconstructed_mask * results[0] + (1 - reconstructed_mask) * sy_np
    image_suffixes = ['reconstructed_image', 'reconstructed_motif']
    paths ={}
    for idx, image in enumerate([reconstructed_image, reconstructed_motif]):
        if image is not None and idx < len(image_suffixes):
            paths[image_suffixes[idx]] = save_numpy_image(image[0], image_suffixes[idx], file_path)
    return paths["reconstructed_image"]



if __name__ == '__main__':
    test('/run/media/white/F6D8A1C6D8A18589/12/123/1.png','/data/code/python/diandi/vmtest/checkpoints/long-text')
