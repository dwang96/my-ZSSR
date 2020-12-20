from PIL import Image, ImageFilter
import tqdm
import math
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import transforms as T
import time
import random
import model
from data import *
from imgAugmentation import *
from config import config
from utils import *

def train(lres_img,
          model,
          data_sampler,
          num_batches,
          sf,
          device):
    
    s_factor = sf # config["s_factor"] # scale factor, defualt 2
    device = config["device"] # cuda
    crop_size = config["crop_size"]
    im_input = lres_img
    input_downscale = bicubic_resample(im_input, 1.0/s_factor)
    interpolated_input_son = bicubic_resample(input_downscale, s_factor, im_input)
    input_son = transforms.ToTensor()(interpolated_input_son)
    average_loss = 0
    loss = nn.MSELoss() # MSE Loss
    learning_rate = config["learning_rate"] # initial learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    run_adjust = 50
    run_test = True
    run_test_every = 50
    # Params concerning learning rate policy
    learning_rate_change_ratio = 1.5  # ratio between STD and slope of linear fit, under which lr is reduced
    learning_rate_policy_check_every = 60
    learning_rate_slope_range = 256
    learning_rate_change_iter_nums = [0]
    min_iters = 256
    
    # start iteration
    mse_reconstruct = []
    mse_steps = []
    # iter = 0
    with tqdm.tqdm(total=num_batches, miniters=1, mininterval=0) as progress:
        for iter, hres in enumerate(data_sampler):
            optimizer.zero_grad()
            lres = hres_to_lres(hres, s_factor)
            lres = bicubic_resample(lres, s_factor, hres)
            # some tricks
            lres = lres.filter(ImageFilter.GaussianBlur(radius=4))
            hres = hres.filter(ImageFilter.SHARPEN)

            trans = transforms.Compose([
                RandomRotationFromSequence([0, 90, 180, 270]),
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
                ToTensor(),
                RandomCrop(crop_size)
            ])
            hres, lres = trans((hres, lres))
                
            hres, lres = hres.unsqueeze(0).to(device), lres.unsqueeze(0).to(device)
            hres_predicted = model(lres)

            error = loss(hres_predicted, hres)

            average_loss += error.item()
            error.backward()
            optimizer.step()
            
            # adjust learning rate
            if run_test and (not iter % run_test_every):
                reconstructed_im = forward(input_son.unsqueeze(0).cuda(), model)
                mse_reconstruct.append(np.mean(np.ndarray.flatten(np.square(im_input - reconstructed_im))))
                mse_steps.append(iter)
            if (not (1 + iter) % learning_rate_policy_check_every and iter - learning_rate_change_iter_nums[-1] > min_iters):
                # noinspection PyTupleAssignmentBalance
                #print(self.conf.run_test_every)
                [slope, _], [[var, _], _] = np.polyfit(mse_steps[-(learning_rate_slope_range //
                                                                        run_adjust):],
                                                        mse_reconstruct[-(learning_rate_slope_range //
                                                                        run_adjust):],
                                                        1, cov=True)

                # We take the the standard deviation as a measure
                std = np.sqrt(var)

#                     # Verbose
#                     print('slope: ', slope, 'STD: ', std)

                # Determine learning rate maintaining or reduction by the ration between slope and noise
                if -learning_rate_change_ratio * slope < std:
                    learning_rate /= 10
                    adjust_learning_rate(optimizer, learning_rate)
                    print("learning rate updated: ", learning_rate)

                    # Keep track of learning rate changes for plotting purposes
                    learning_rate_change_iter_nums.append(iter)

            if iter > num_batches or learning_rate < 9e-6:
                print('Training is over.')
                average_loss /= iter
                print('Average Loss is {average_loss}'.format(average_loss=average_loss))
#                 out = final_output(im_input, model, resample, s_factor, back_projection)
#                 final_zssr.save('zssr.png')
                break

            progress.set_description("Iteration: {iter} Loss: {loss}, Learning Rate: {lr} ".format( \
               iter=iter, loss=error.data.cpu().numpy(), lr=learning_rate))
            progress.update()
            
            

            
def hres_to_lres(hres, sf):

    """
    generate corresponding lres_son base on the given scale factor.
    """
    lres = bicubic_resample(hres, 1.0 / sf)
    return lres

def final_output(im_input, model, resample_method, sf, back_projection):

    outputs = []
    # im_input = im_input.filter(ImageFilter.SHARPEN)

    for k in range(0, 8):
        test_input = np.rot90(im_input, k) if k < 4 else np.fliplr(np.rot90(im_input, k))
        test_input = Image.fromarray(test_input)
        interpolated_test_input = resample_method(test_input, sf)
        interpolated_test_input = transforms.ToTensor()(interpolated_test_input).unsqueeze(0).cuda()
        temp_out = forward(interpolated_test_input, model)
        
        

        temp_out = np.rot90(temp_out, -k) if k < 4 else np.rot90(np.fliplr(temp_out), -k)
        temp_out = temp_out.copy()
        temp_out = transforms.ToTensor()(temp_out)
        temp_out = transforms.ToPILImage()(temp_out)
        # temp_out.save('zssr_init_'+k)

        for bp_iter in range(12):
            temp_out = back_projection(temp_out, im_input, sf, resample_method)

        temp_out.save('./test_result/zssr_init_'+'%d' % k +'.png')
        temp_out = np.array(temp_out)
        outputs.append(temp_out)
        

    out_median = np.clip(np.median(outputs, 0) / 255, 0, 1)
    out_median = transforms.ToTensor()(out_median)
    out_median = transforms.ToPILImage()(out_median)
    out_median.save('./test_result/out_median.png')
    # im_input = im_input.filter(ImageFilter.SHARPEN)
    for bp_iter in range(10):
        out_median = back_projection(out_median, im_input, sf, resample_method)

    return out_median, outputs

    