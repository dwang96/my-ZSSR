import math
import numpy as np
from torchvision.transforms import transforms
from PIL import Image, ImageChops
from config import config
from skimage import transform


def psnr_img(img, reconstructed_img):
    rmse = math.sqrt(np.mean((img - reconstructed_img) ** 2))
    max_val = img.max()
    return 20 * math.log10(max_val / rmse)

def bicubic_resample(lres, sf, hres=None):
    if hres is None:
        lres_inter = lres.resize([round(lres.size[0] * sf), round(lres.size[1] * sf)], resample=Image.BICUBIC)
        return lres_inter
    else:
        lres_inter = lres.resize([round(lres.size[0] * sf), round(lres.size[1] * sf)], resample=Image.BICUBIC)
        if lres_inter.size[0] == hres.size[0] and lres_inter.size[1] == hres.size[1]:
            return lres_inter
        else: 
            lres_inter = lres.resize([hres.size[0], hres.size[1]], resample=Image.BICUBIC)
            return lres_inter
    
def adjust_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def forward(lres_interpolated, model):
    hres_forward = model(lres_interpolated)
    
    hres_cpu = hres_forward.cpu().detach().permute(0, 2, 3, 1).numpy()
    # print(hres_cpu.shape)

    return np.clip(np.squeeze(hres_cpu), 0, 1)

def back_projection(hres_out, lres_in, sf, resample_method):
    lres_from_hres = resample_method(hres_out, 1.0/sf, lres_in)
    lres_from_hres_tensor = transforms.ToTensor()(lres_from_hres)
    lres_in_tensor = transforms.ToTensor()(lres_in)
    residual_back_projection = lres_in_tensor - lres_from_hres_tensor
    residual_back_projection = residual_back_projection.permute(1, 2, 0).numpy()

    residual_out = transform.resize(residual_back_projection,(hres_out.size[1], hres_out.size[0]), order=3)
    
    residual_out_tensor = transforms.ToTensor()(residual_out)
    hres_out_tensor = transforms.ToTensor()(hres_out)

    hres_final_tensor = residual_out_tensor + hres_out_tensor

    hres_final = np.clip(hres_final_tensor.numpy(), 0, 1)
    hres_final_tensor = transforms.ToTensor()(hres_final).permute(1, 2, 0)
    hres_final = transforms.ToPILImage()(hres_final_tensor)

    return hres_final

