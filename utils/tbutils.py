import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision


def get_colormap(N=256):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3
        cmap[i] = np.array([r, g, b])
    return cmap


COLORMAP = get_colormap()


def denormalize_img(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    out = torch.zeros_like(img)
    for i in range(3):
        out[:, i, :, :] = img[:, i, :, :] * std[i] + mean[i]
    out = (out * 255).to(torch.uint8)
    return out


def make_grid_image(img, cam, cls_label, nrow=2, mask=None):
    cam = cam.detach().cpu()
    img = img.detach().cpu()
    cls_label = cls_label.detach().cpu()

    img = denormalize_img(img)
    if mask is None:
        grid_img = torchvision.utils.make_grid(img, nrow=nrow)
    else:
        mask = mask.detach().cpu()
        mask = F.interpolate(mask.unsqueeze(1).float(), size=img.shape[-2:], mode="nearest")
        grid_img = torchvision.utils.make_grid((img * 0.5 + mask * 255 * 0.5).to(torch.uint8), nrow=nrow)

    cam = F.interpolate(cam, size=img.shape[2:], mode="bilinear", align_corners=False)
    cam = cam * cls_label.unsqueeze(2).unsqueeze(3)
    cam_max = torch.max(cam, dim=1)[0].numpy()

    cam = plt.get_cmap("jet")(cam_max)[:, :, :, :3] * 255
    cam = torch.from_numpy(cam).permute(0, 3, 1, 2)
    cam = (cam * 0.5 + img * 0.5).to(torch.uint8)
    grid_cam = torchvision.utils.make_grid(cam, nrow=nrow)

    return grid_img, grid_cam

def make_grid_image_bkg(img, cam, cls_label, nrow=2, mask=None):
    cam = cam.detach().cpu()
    img = img.detach().cpu()
    cls_label = cls_label.detach().cpu()

    img = denormalize_img(img)
    if mask is None:
        grid_img = torchvision.utils.make_grid(img, nrow=nrow)
    else:
        mask = mask.detach().cpu()
        mask = F.interpolate(mask.unsqueeze(1).float(), size=img.shape[-2:], mode="nearest")
        grid_img = torchvision.utils.make_grid((img * 0.5 + mask * 255 * 0.5).to(torch.uint8), nrow=nrow)

    cam = F.interpolate(cam, size=img.shape[2:], mode="bilinear", align_corners=False)
    # cam = cam * cls_label.unsqueeze(2).unsqueeze(3)
    cam_max = torch.max(cam, dim=1)[0].numpy()

    cam = plt.get_cmap("jet")(cam_max)[:, :, :, :3] * 255
    cam = torch.from_numpy(cam).permute(0, 3, 1, 2)
    cam = (cam * 0.5 + img * 0.5).to(torch.uint8)
    grid_cam = torchvision.utils.make_grid(cam, nrow=nrow)

    return grid_img, grid_cam


def make_grid_label(label, nrow=2):
    label = label.cpu().numpy().astype(int)
    label_cmap = COLORMAP[label, :]
    label_cmap = torch.from_numpy(label_cmap).permute(0, 3, 1, 2)
    grid_label = torchvision.utils.make_grid(label_cmap, nrow=nrow)
    return grid_label


def min_max_norm(x, eps=1e-8):
    shape = x.shape
    x = x.reshape(shape[0], -1)
    x_min = torch.min(x, dim=1, keepdim=True)[0]
    x_max = torch.max(x, dim=1, keepdim=True)[0]
    x = (x - x_min) / (x_max - x_min + eps)
    x = x.reshape(shape)
    return x


def make_grid_attention(attention, nrow=2, B=2):
    attn = attention.detach().mean(1)[:B, 1:, 1:]
    attn = min_max_norm(attn)
    attn = attn.cpu().numpy()
    cmap = plt.get_cmap("Reds")(attn)[:, :, :, :3] * 255
    cmap = torch.from_numpy(cmap).permute(0, 3, 1, 2)
    grid = torchvision.utils.make_grid(cmap.to(torch.uint8), nrow=nrow)
    return grid
