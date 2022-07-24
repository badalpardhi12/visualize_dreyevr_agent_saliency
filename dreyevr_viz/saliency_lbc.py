# Visualizing and Understanding Atari Agents | Sam Greydanus | 2017 | MIT License

from __future__ import print_function
import warnings ; warnings.filterwarnings('ignore') # mute warnings, live dangerously ;)

import torch
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
from scipy.ndimage.filters import gaussian_filter
# from scipy.misc import imresize # preserves single-pixel info _unlike_ img = img[::2,::2]
import cv2

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
    """
    mask to blur 
    """
    y,x = np.ogrid[-center[0]:size[0]-center[0], -center[1]:size[1]-center[1]]
    keep = x*x + y*y <= 1
    mask = np.zeros(size) ; mask[keep] = 1 # select a circle of pixels
    mask = gaussian_filter(mask, sigma=r) # blur the circle of pixels. this is a 2D Gaussian for r=r^2=1
    return mask/mask.max()

def run_through_model(model, history, ix, interp_func=None, mask=None, blur_memory=None, mode='actor'):
    if mask is None:
        im = prepro(history['ins'][ix])
    else:
        assert(interp_func is not None, "interp func cannot be none")
        im = interp_func(prepro(history['ins'][ix]).squeeze(), mask).reshape(1,post_h,post_w) # perturb input I -> I'
    tens_state = torch.Tensor(im)
    state = Variable(tens_state.unsqueeze(0), volatile=True)
    hx = Variable(torch.Tensor(history['hx'][ix-1]).view(1,-1))
    cx = Variable(torch.Tensor(history['cx'][ix-1]).view(1,-1))
    if blur_memory is not None: cx.mul_(1-blur_memory) # perturb memory vector
    return model((state, (hx, cx)))[0] if mode == 'critic' else model((state, (hx, cx)))[1]

def score_frame(model, history, ix, r, d, interp_func, mode='actor'):
    # r: radius of blur
    # d: density of scores (if d==1, then get a score for every pixel...
    #    if d==2 then every other, which is 25% of total pixels for a 2D image)
    assert mode in ['actor', 'critic'], 'mode must be either "actor" or "critic"'
    L = run_through_model(model, history, ix, interp_func, mask=None, mode=mode)
    scores = np.zeros((int(post_h/d)+1,int(post_w/d)+1)) # saliency scores S(t,i,j)
    for i in range(0,post_h,d):
        for j in range(0,post_w,d):
            mask = get_mask(center=[i,j], size=[post_h,post_w], r=r)
            l = run_through_model(model, history, ix, interp_func, mask=mask, mode=mode)
            scores[int(i/d),int(j/d)] = (L-l).pow(2).sum().mul_(.5).data.item()
    # avoid range artifacts while resizing
    pmax = scores.max()
    scores = cv2.resize(scores, dsize=(post_h,post_w), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return pmax * scores / scores.max()

def saliency_on_atari_frame(saliency, atari, fudge_factor, channel=2, sigma=0):
    # sometimes saliency maps are a bit clearer if you blur them
    # slightly...sigma adjusts the radius of that blur
    pmax = saliency.max()
    S = cv2.resize(saliency, dsize=(img_h,img_w), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    S = S if sigma == 0 else gaussian_filter(S, sigma=sigma)
    S -= S.min() ; S = fudge_factor*pmax * S / S.max()
    I = atari.astype('uint16')
    # TODO: this 35:195 height crop is atari specific and we prolly don't need it
    I[35:195,:,channel] += S.astype('uint16') 
    I = I.clip(1,255).astype('uint8')
    return I

def get_env_meta(env_name):
    meta = {}
    if env_name=="Pong-v0":
        meta['critic_ff'] = 600 ; meta['actor_ff'] = 500
    elif env_name=="Breakout-v0":
        meta['critic_ff'] = 600 ; meta['actor_ff'] = 300
    elif env_name=="SpaceInvaders-v0":
        meta['critic_ff'] = 400 ; meta['actor_ff'] = 400
    else:
        print('environment "{}" not supported'.format(env_name))
    return meta