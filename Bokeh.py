import cv2
import numpy as np
from PIL import Image
from skimage.morphology import disk, diamond, square
from skimage.filters.rank import mean
from skimage.color.adapt_rgb import adapt_rgb, each_channel
import torch
import torchvision
import sys
import os.path
import warnings

BLUR_TYPES = ['blur', 'diamond', 'square']

def make_kernel_from_shape(shape):
    kernel = cv2.getGaussianKernel(11, 5.)
    kernel = kernel * kernel.transpose() * shape 
    kernel_normalized = kernel / np.sum(kernel)
    return kernel_normalized

def get_subject_mask(image):

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    image = image/255
    t = torch.from_numpy(image).float()
    t = t.permute(2,0,1)
    prediction = model([t])[0]
    mask = prediction['masks'].detach().numpy()[0][0]

    return mask


#OPTED TO NOT USE THIS AS IT WAS INCONSISTENT 
def graph_cut(image, mask):

    # make mask more general to make it easier for the graph cut algorithm to properly segement the subject
    helper_mask = mask.copy()
    helper_mask[helper_mask > 0.10] = 1 
    helper_mask = helper_mask.astype('uint8')
    
    #Setting values for mask 
    helper_mask[helper_mask == 1] = cv2.GC_PR_FGD
    helper_mask[helper_mask == 0] = cv2.GC_BGD
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    
    fine_mask, bgdModel, fgdModel = cv2.grabCut(image, helper_mask, None, 
                                               bgdModel, fgdModel, 5, mode=cv2.GC_INIT_WITH_MASK)
    
    fine_mask = np.where((fine_mask==2)| (fine_mask==0), 0, 1).astype('uint8')
    return fine_mask

@adapt_rgb(each_channel)
def apply_bokeh(image, shape, blur_strength):
    if shape == 'diamond':
        kernel = diamond(blur_strength)
    elif shape == 'square':
        kernel = square(blur_strength)
    else:
        kernel = disk(blur_strength)
       
    return mean(image, kernel)


def main():
    
   
    if len(sys.argv) != 4:
        print("Enter 3 Arguments only: image name as string, blur type(diamond, blur, square), and blur strength")
        return
    

    image_path = sys.argv[1]
   
    if not os.path.isfile(image_path):
        print('Error: file does not exist')
        return

    blur_type = sys.argv[2]

    if blur_type not in BLUR_TYPES:
        print('Error: Blur options are only [diamond, blur, square]')
    

    blur_strength = int(sys.argv[3])

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    #get the subject mask
    print("detecting subject in image...")
    subject_mask = get_subject_mask(image)
    print("retrieved subject")
    

    #get background
    bg_mask = np.abs(1 - subject_mask)
    background = image*bg_mask[:,:,np.newaxis]
    background = np.asarray(background, dtype='uint8')

    #apply morophology to remove some noise from the subject mask
    kernel = np.ones((5,5), np.uint8)
    subject_mask = cv2.erode(subject_mask, kernel, iterations= 2)

    #smooth the mask a little as a form of anti aliasing
    subject_mask = cv2.GaussianBlur(subject_mask, (1,1), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_DEFAULT)

    #retrieve the subject from the image
    subject = np.asarray(image*subject_mask[:,:,np.newaxis], dtype='uint8')

    #apply bokeh to the background
    print("emulating bokeh effect")
    bokeh = np.asarray(apply_bokeh(background, blur_type, blur_strength), dtype='uint8')
    print("finished emulating effect")

    #add both the foreground and background together
    result = cv2.addWeighted(subject, 1, bokeh, 1, 0)

    cv2.imwrite(image_path.replace(".png", "_Bokeh.png"), cv2.cvtColor(result, cv2.COLOR_BGR2RGB))



if __name__ == "__main__":
    main()
