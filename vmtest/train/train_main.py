from utils.train_utils import *
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# paths
root_path = '..'
train_tag = 'text_color'


# datasets paths
cache_root = ['data folder a', 'data folder b', '...']
cache_root = [

              '/run/media/white/F6D8A1C6D8A18589/test/pdf_big_gray_bound_blur',
    '/run/media/white/F6D8A1C6D8A18589/test/pdf_big_gray',

    '/run/media/white/F6D8A1C6D8A18589/test/danju_big_gray',
    '/run/media/white/F6D8A1C6D8A18589/test/danju_big_gray_bound_blur',
    '/run/media/white/F6D8A1C6D8A18589/test/danju_big_color_bound_blur',
    '/run/media/white/F6D8A1C6D8A18589/test/shopping_big_gray',
    '/run/media/white/F6D8A1C6D8A18589/test/shopping_big_color',

    '/run/media/white/F6D8A1C6D8A18589/test/shopping_big_color_bound_blur',
    # '/run/media/white/F6D8A1C6D8A18589/test/shopping_water',
    '/run/media/white/F6D8A1C6D8A18589/test/pdf_big_color_bound_blur',
    '/run/media/white/F6D8A1C6D8A18589/test/coco_big_color',
    '/run/media/white/F6D8A1C6D8A18589/test/coco_big_gray',
    '/run/media/white/F6D8A1C6D8A18589/test/shopping_big_color_notrans',


    # '/run/media/white/F6D8A1C6D8A18589/test/txt_coco_color',
    # '/run/media/white/F6D8A1C6D8A18589/test/txt_coco_gray',
    # '/run/media/white/F6D8A1C6D8A18589/test/txt_danju_color',
    # '/run/media/white/F6D8A1C6D8A18589/test/txt_danju_gray',
    # '/run/media/white/F6D8A1C6D8A18589/test/txt_shape_shopping_color',
    # '/run/media/white/F6D8A1C6D8A18589/test/txt_shopping_color',
    # '/run/media/white/F6D8A1C6D8A18589/test/txt_shopping_gray',
    # '/run/media/white/F6D8A1C6D8A18589/test/txt_shopping_white',
    # '/run/media/white/F6D8A1C6D8A18589/test/txt_coco_moji',



              ]

# dataset configurations

image_size = 512
patch_size = False
resize_size = 256

# patch_size = 128
# resize_size = False

# network
nets_path = '%s/checkpoints/%s' % (root_path, train_tag)
images_path = '%s/images' % nets_path

num_blocks = (3, 3, 3, 3, 3)
shared_depth = 2
use_vm_decoder = True

log_dir = os.path.join(root_path, 'logs')
log = SummaryWriter(log_dir=log_dir)

# train configurations
gamma1 = 2   # L1 image
gamma2 = 1   # L1 visual motif
epochs = 10000
batch_size = 3
print_frequency = 20
save_frequency = 2
device = torch.device('cuda:0')


def l1_relative(reconstructed, real, batch, area):
    loss_l1 = torch.abs(reconstructed - real).view(batch, -1)
    loss_l1 = torch.sum(loss_l1, dim=1) / area
    loss_l1 = torch.sum(loss_l1) / batch
    return loss_l1


def train(net, train_loader, test_loader):
    bce = nn.BCELoss()
    net.set_optimizers()
    losses = []
    print('Training Begins')
    total_step = len(train_loader)
    for epoch in range(epochs):
        real_epoch = epoch + 1
        for i, data in enumerate(train_loader, 0):
            synthesized, images, vm_mask, motifs, vm_area = data
            synthesized, images, = synthesized.to(device), images.to(device)
            vm_mask, vm_area = vm_mask.to(device), vm_area.to(device)
            results = net(synthesized)
            guess_images, guess_mask = results[0], results[1]
            expanded_vm_mask = vm_mask.repeat(1, 3, 1, 1)
            reconstructed_pixels = guess_images * expanded_vm_mask
            real_pixels = images * expanded_vm_mask
            batch_cur_size = vm_mask.shape[0]
            net.zero_grad_all()
            loss_l1_images = l1_relative(reconstructed_pixels, real_pixels, batch_cur_size, vm_area)
            loss_mask = bce(guess_mask, vm_mask)
            loss_l1_vm = 0
            if len(results) == 3:
                guess_vm = results[2]
                reconstructed_motifs = guess_vm * expanded_vm_mask
                real_vm = motifs.to(device) * expanded_vm_mask
                loss_l1_vm = l1_relative(reconstructed_motifs, real_vm, batch_cur_size, vm_area)
            loss = gamma1 * loss_l1_images + gamma2 * loss_l1_vm + loss_mask
            loss.backward()
            net.step_all()
            losses.append(loss.item())
            # print
            if (i + 1) % print_frequency == 0:
                lavg = sum(losses) / len(losses)
                print('%s [%d, %3d] , baseline loss: %.2f' % (train_tag, real_epoch, batch_size * (i + 1), lavg))
                losses = []
                log.add_scalar('loss',  lavg, i + epoch * total_step + 1)
                info = {'synthesized':
                            transform_to_numpy_image(synthesized),
                        'images_real':
                            transform_to_numpy_image(images),
                        'images_guess':
                            transform_to_numpy_image(guess_images),
                        'image_reconstructed':
                            transform_to_numpy_image(reconstructed_pixels),
                        'vm_real':
                            transform_to_numpy_image(real_vm),
                        'vm_guess':
                            transform_to_numpy_image(guess_vm),
                        'mask_guess':
                            transform_to_numpy_image(guess_mask),
                        'reconstructed_motifs':
                            transform_to_numpy_image(reconstructed_motifs),
                        }
                for tag, imgs in info.items():
                    log.add_images(tag, imgs, i + epoch * total_step + 1, dataformats='NHWC')
        # savings
        if real_epoch % save_frequency == 0:
            print("checkpointing...")
            # image_name = '%s/%s_%d.png' % (images_path, train_tag, real_epoch)
            # _ = save_test_images(net, test_loader, image_name, device)
            torch.save(net.state_dict(), '%s/net_baseline.pth' % nets_path)
            torch.save(net.state_dict(), '%s/net_baseline_%d.pth' % (nets_path, real_epoch))
            print("done")

    print('Training Done:)')
def transform_to_numpy_image(tensor_image):
    image = tensor_image.cpu().detach().numpy()
    image = np.transpose(image, (0, 2, 3, 1))
    if image.shape[3] != 3:
        image = np.repeat(image, 3, axis=3)
    else:
        image = (image / 2 + 0.5)
    images = (image * 255).astype(np.uint8)
    return image

def run():
    init_folders(nets_path, images_path)
    opt = load_globals(nets_path, globals(), override=True)
    train_loader, test_loader = init_loaders(opt, cache_root=cache_root)
    base_net = init_nets(opt, nets_path, device)
    # i = 0;
    # for param in base_net.parameters():
    #     print(param)
    #     i = i+1
    #     if i>3:
    #         break
    # for param in base_net.parameters():
    #     param = param+0.2
    # for param in base_net.parameters():
    #     print(param)
    train(base_net, train_loader, test_loader)


if __name__ == '__main__':
    run()
