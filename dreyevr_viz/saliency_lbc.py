# Visualizing and Understanding Atari Agents | Sam Greydanus | 2017 | MIT License

from __future__ import print_function
import warnings ; warnings.filterwarnings('ignore') # mute warnings, live dangerously ;)

import torch
import torchvision

import numpy as np
from scipy.ndimage.filters import gaussian_filter
import cv2
from copy import deepcopy

post_h = 144
post_w = 256

# this is crop the bottom and top scores, then avg across channel to grayscale and then normalize
# prepro = lambda img: cv2.resize(img[35:195].mean(2), (post_h,post_w)).astype(np.float32).reshape(1,post_h,post_w)/255.
# searchlight = lambda I, mask: I*mask + gaussian_filter(I, sigma=3)*(1-mask) # choose an area NOT to blur
# occlude = lambda I, mask: I*(1-mask) + gaussian_filter(I, sigma=3)*mask # choose an area to blur
def prepro(img):
    return img.astype(np.float32)/255.

def occlude(I, mask):
    return I*(1-mask) + gaussian_filter(I, sigma=3)*mask



def get_mask(center, size, r):
    y,x = np.ogrid[-center[0]:size[0]-center[0], -center[1]:size[1]-center[1]]
    keep = x*x + y*y <= 1
    mask = np.zeros(size) ; mask[keep] = 1 # select a circle of pixels
    mask = gaussian_filter(mask, sigma=r) # blur the circle of pixels. this is a 2D Gaussian for r=r^2=1
    return mask/mask.max()

def apply_mask(input_data, mask, interp_func, channel=0):
    masked_data = deepcopy(input_data)
    if channel < 3:        
        img = masked_data['image'][..., (channel*3):(channel*3)+3]
        # perturb input I -> I'
        im1 = interp_func(prepro(img[...,0]).squeeze(), mask)
        im2 = interp_func(prepro(img[...,1]).squeeze(), mask)
        im3 = interp_func(prepro(img[...,2]).squeeze(), mask)
        masked_data['image'][..., (channel*3):(channel*3)+3] = (np.stack((im1, im2, im3), axis=2)*255).astype(int)
    elif channel == 3: 
        img = masked_data['target_img']
        # no prepro reqd -- already done
        im = interp_func((img).squeeze(), mask).reshape(post_h,post_w).astype(np.float32)
        masked_data['target_img'] = im[None]
    else:
        raise ValueError("channel must be 0-2:rgb/left/right or 3:command heatmap")     
    return masked_data

def _get_masked_data_parallel(ijk, other_data):
    i,j,k = ijk
    input_data, r, interp_func = other_data 
    return get_and_apply_mask([i,j], input_data, r, interp_func, k)

def get_and_apply_mask(center, input_data, r, interp_func, channel):
    H, W = input_data['image'][..., 0].shape[:2]
    mask = get_mask(center=center, size=[H, W], r=r)
    masked_data = apply_mask(input_data, mask, interp_func=occlude, channel=channel)
    return masked_data

# if computing salience of "target_heatmap" -- incorporates commands

def fwd_img_target_model(dreyevr_img_agent, input_data):
#     tick_data = dreyevr_img_agent.offline_tick(input_data)
    tick_data = deepcopy(input_data)
    img = torchvision.transforms.functional.to_tensor(tick_data['image'])
    img = img[None].cuda()
    target_heatmap_cam = torch.from_numpy(input_data['target_img'])[None].cuda()

    out, logits = dreyevr_img_agent.net.net(torch.cat((img, target_heatmap_cam), 1), True)
    # this is (1x4xHW) -- 4 is the num of intermediate pts being pred 
    flat_logits = logits.view(logits.shape[:-2] + (-1,)) #.detach().cpu().numpy()
    return flat_logits

def fwd_img_target_model_batch(dreyevr_img_agent, batch_data):
#     tick_data = dreyevr_img_agent.offline_tick(input_data)
    # order is 'rgb', 'rgb_left', 'rgb_right'
    result = [torchvision.transforms.functional.to_tensor(input_data_dict['image'])\
                       for input_data_dict in batch_data]
    imgs_tensor = torch.stack(result, 0)
    imgs_tensor = imgs_tensor.cuda()

    # batch_data['target_img']
    target_imgs_tensor = [torch.from_numpy(input_data_dict['target_img'])\
                       for input_data_dict in batch_data]
    target_imgs_tensor = torch.stack(target_imgs_tensor, 0)
    target_imgs_tensor = target_imgs_tensor.cuda()

    out, logits = dreyevr_img_agent.net.net(torch.cat((imgs_tensor, target_imgs_tensor), 1), True)
    # this is (1x4xHW) -- 4 is the num of intermediate pts being pred 
    flat_logits = logits.view(logits.shape[:-2] + (-1,)) #.detach().cpu().numpy()
    return flat_logits

# if not computing salience of "target_heatmap" -- incorporates commands

def fwd_img_model(dreyevr_img_agent, input_data):
#     tick_data = dreyevr_img_agent.offline_tick(input_data)
    # tick_data = deepcopy(input_data)
    img = torchvision.transforms.functional.to_tensor(tick_data['image'])
    img = img[None].cuda()

    target = torch.from_numpy(tick_data['target'])
    target = target[None].cuda()
    _, (_, _), logits = dreyevr_img_agent.net.forward_w_logit(img, target)
    # this is (1x4xHW) -- 4 is the num of intermediate pts being pred 
    flat_logits = logits.view(logits.shape[:-2] + (-1,)) #.detach().cpu().numpy()
    return flat_logits

def fwd_img_model_batch(dreyevr_img_agent, batch_data):
    # img = torchvision.transforms.functional.to_tensor(tick_data[1:]['image'])    
    result = [torchvision.transforms.functional.to_tensor(input_data_dict['image'])\
                       for input_data_dict in batch_data]
    imgs_tensor = torch.stack(result, 0)
    imgs_tensor = imgs_tensor.cuda()

    targets_tensor = [torch.from_numpy(input_data_dict['target'])\
                       for input_data_dict in batch_data]
    targets_tensor = torch.stack(targets_tensor,0)
    targets_tensor = targets_tensor.cuda()

    _, (_, _), logits = dreyevr_img_agent.net.forward_w_logit(imgs_tensor, targets_tensor)
    flat_logits = logits.view(logits.shape[:-2] + (-1,))
    return flat_logits



def score_frame(dreyevr_img_agent, input_data, r, d, interp_func, pt_aggregate="leading"):
    # r: radius of blur
    # d: density of scores (if d==1, then get a score for every pixel...
    #    if d==2 then every other, which is 25% of total pixels for a 2D image)
    # unmodified image logits
    input_data = dreyevr_img_agent.offline_tick(input_data)
    L = fwd_img_model(dreyevr_img_agent, input_data)    
    scores = np.zeros((int(post_h/d)+1,int(post_w/d)+1, 3)) # saliency scores S(t,i,j)

    for i in range(0,post_h,d):
        for j in range(0,post_w,d):
            for k in range(0,3): # this is for the channel rgb/left/right
                mask = get_mask(center=[i,j], size=[post_h,post_w], r=r)
                masked_data = apply_mask(input_data, mask, occlude, channel=k)
                # masked image logits
                l = fwd_img_model(dreyevr_img_agent, masked_data)
                # this corresponds to 
                if pt_aggregate=="leading":
                    scores[int(i/d),int(j/d), k] = (L-l)[:,:2,:].pow(2).sum().mul_(.5).data.item()
                elif pt_aggregate=="all":
                    scores[int(i/d),int(j/d), k] = (L-l).pow(2).sum().mul_(.5).data.item()
                else:
                    raise ValueError("only 'leading'(first 2) and 'all' aggregations are supported")

    pmax = scores.max()
    scores = cv2.resize(scores, dsize=(post_w, post_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    smap = pmax * scores / scores.max()
    smap = smap.astype(int)
    return smap


def score_frame_batched(dreyevr_img_agent, input_data, r, d, interp_func,
             pt_aggregate="leading", batch_size=64, include_target=True,
             return_target_map=False):
    # r: radius of blur
    # d: density of scores (if d==1, then get a score for every pixel...
    #    if d==2 then every other, which is 25% of total pixels for a 2D image)
    # unmodified image logits
    input_data = dreyevr_img_agent.offline_tick(input_data)

    # def fwd_img_cmd_model(dreyevr_img_agent, input_data, mask=None):
    tick_data = dreyevr_img_agent.offline_tick(input_data)
    img = torchvision.transforms.functional.to_tensor(tick_data['image'])[None].cuda()
    target = torch.from_numpy(tick_data['target'])[None].cuda()    
    # get the unomdified target image
    target_heatmap_cam = dreyevr_img_agent.net.target_to_target_heatmap(img, target)
    target_img_OG = target_heatmap_cam.detach().cpu().numpy().squeeze()
    input_data['target_img'] = target_img_OG[None]
    
    L = fwd_img_target_model(dreyevr_img_agent, input_data)
    
    # TODO could do more parallelism here -- batch apply the gaussian blur,
    # generate masks, then Hadamard product to generate the masked imgs

    num_max_channels = 4 if include_target else 3

    masked_data_arr = np.empty((int(post_h/d)+1, int(post_w/d)+1, num_max_channels), dtype=dict)    

    for i in range(0,post_h,d):
        for j in range(0,post_w,d):
            for k in range(0, num_max_channels): # this is for the channel rgb/left/right
                masked_data_arr[int(i/d),int(j/d), k] = get_and_apply_mask([i,j], input_data, r, interp_func, k)
    masked_data_flat = masked_data_arr.reshape(-1)

    # aggregate batches for forward
    num_batches = int(masked_data_flat.size/batch_size)
    num_batches = (num_batches+1) if masked_data_flat.size%batch_size else num_batches
    scores = np.zeros(shape=masked_data_flat.shape)

    for i in range(num_batches):
        if i < num_batches-1:
            batch_data = masked_data_flat[i*batch_size:(i+1)*batch_size]        
        else:
            batch_data = masked_data_flat[i*batch_size:]    

        if include_target:
            flat_logits = fwd_img_target_model_batch(dreyevr_img_agent, batch_data)
        else:
            flat_logits = fwd_img_model_batch(dreyevr_img_agent, batch_data)

        if pt_aggregate=="leading":
            score_temp = (L-flat_logits)[:,:2,:].pow(2).sum(dim=[1,2]).mul_(.5).data.tolist()
        else:
            score_temp = (L-flat_logits).pow(2).sum(dim=[1,2]).mul_(.5).data.tolist()

        if i< num_batches-1:
            scores[i*batch_size:(i+1)*batch_size] = score_temp
        else:
            scores[i*batch_size:] = score_temp
    scores = scores.reshape(masked_data_arr.shape)    
#     ijks = list(itertools.product(*[range(0,post_h,d), range(0,post_w,d), range(3)]))
#     other_data = (input_data, r, interp_func)
#     _args = zip(ijks, itertools.repeat(other_data))
#     with Pool() as pool:
#         results = pool.starmap(_get_masked_data_parallel, _args)
    
#     results  
    pmax = scores.max()
    scores = cv2.resize(scores, dsize=(post_w, post_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    smap = pmax * scores / scores.max()
    smap = smap.astype(int)
    if return_target_map:
        return smap, target_img_OG
    # else
    return smap