from PIL import Image
import numpy as np
import cv2
import scipy.stats as stats
import matplotlib.pyplot as plt
# A function to get a gaussian blob of image size centered at x,y

def get_gaussian_blob(x, y, sigma, img_size, radius):
    attentionMap = np.zeros(img_size) # 2D array of zeros
    # check if x and y are within the image size
    if x < 0 or y < 0 or x >= img_size[0] or y >= img_size[1]:
        return attentionMap

    # create a mask of size radius centered at x,y
    A, B = np.ogrid[:img_size[0], :img_size[1]]
    dist_from_center = np.sqrt((A - x) ** 2 + (B - y) ** 2)
    mask = dist_from_center <= radius
    
    # create gaussian kernel
    size = 2 * radius
    X = np.linspace(-sigma, sigma, size + 1)
    Y = np.diff(stats.norm.cdf(X))
    kernel = np.outer(Y, Y)
    kernel = kernel / kernel.sum()

    # apply the kernel to the attention map
    xMin = max(0, x - radius)
    xMax = min(img_size[0], x + radius)
    yMin = max(0, y - radius)
    yMax = min(img_size[1], y + radius)

    xMinKernel = max(0, radius - x)
    xMaxKernel = min(size, radius + img_size[0] - x)
    yMinKernel = max(0, radius - y)
    yMaxKernel = min(size, radius + img_size[1] - y)

    attentionMap[xMin:xMax, yMin:yMax] = kernel[xMinKernel:xMaxKernel, yMinKernel:yMaxKernel]
    
    # set all the values outside the mask to 0
    attentionMap[~mask] = 0    
    
    return attentionMap

# A function that takes in an array of gaze points and returns an attention map
def get_attention_map(gazeCoords, imgShape, sigma=1, radius = 1):
    '''
    Here the gazeCoords corresponds to all the gaze points 
    '''
    # create a zero array of the size of the image
    attentionMap = np.zeros(imgShape)
    # for each gaze point, add a gaussian blob to the attention map
    for gazePoint in gazeCoords:
        # get the x and y coordinates of the gaze point
        x, y = gazePoint
        # add a gaussian blob to the attention map
        attentionMap += get_gaussian_blob(x, y, sigma, imgShape, radius=radius)
    # return the attention map
    return attentionMap
    
# a function to draw the attention map on the image
def draw_attention_map(gazeCoords, 
                                        img, 
                                        alpha=0.3, 
                                        sigma= 5, 
                                        radius = 10, 
                                        image_blur = False, 
                                        glob_blur = True, 
                                        color_= True,
                                        threshold = 0):
    # input is a PIL image
    # convert the image to numpy array
    img = np.array(img, dtype=np.uint8)
    # interchange the x and y coordinates
    gazeCoordsNew = (np.array(gazeCoords)[:, [1, 0]]).astype(int)
    # get the attention map
    attentionMap = get_attention_map(gazeCoordsNew, img.shape[:2], sigma=sigma, radius=radius)
    attentionMap[attentionMap < threshold] = 0

    if image_blur:
        # blur the image except the region of interest
        img[attentionMap == 0] = cv2.blur(img, (7, 7))[attentionMap == 0]
    
    if glob_blur:
        # apply attention map with alpha blending to the image
        if color_:
            # color the attention map to Jet colormap
            attentionMapCopy = attentionMap.copy()
            attentionMapCopy = attentionMapCopy / attentionMapCopy.max()
            # create a colormap image of the attention map using matplotlib
            heatMap = plt.cm.jet(attentionMapCopy)[:, :, :3]
            heatMap = (heatMap * 255).astype(np.uint8)

            # apply the heatMap to the image only where the attention map is greater than 0
            img[attentionMap > 0] = img[attentionMap > 0] * (1 - alpha) + heatMap[attentionMap > 0] * (alpha)
        else:
            img[attentionMap > 0] = img[attentionMap > 0] * (1 - alpha) + attentionMap[attentionMap > 0][:, None] * 255 * (alpha)

    # convert the image back to PIL image
    img = Image.fromarray(img.astype('uint8'))
    return img, attentionMap