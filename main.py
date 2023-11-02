import numpy as np
import torch
import nibabel as nib

""" Input images """ 
IMG_CTA = nib.load("path_to_cta.nii.gz").get_fdata() # shape (146,182,144)
IMG_VASC = nib.load("path_to_3D_vasculature.nii.gz").get_fdata() # shape (146,182,144)


""" ResNet """
# Cases
ResNet_type = 'input_only_brain_CTA' 
# input_only_brain_CTA
# input_only_vasculature
# input_brain_CTA_and_vasculature

image = np.expand_dims(IMG_CTA,0)
imageVasc = np.expand_dims(IMG_VASC,0)

if ResNet_type == 'input_only_brain_CTA':
    while not len(image.shape) == 5:
        image = np.expand_dims(image,0)
        image_torch = torch.tensor(np.ascontiguousarray(image), dtype=torch.float32)

elif ResNet_type == 'input_only_vasculature':
    while not len(imageVasc.shape) == 5:
        image = np.expand_dims(imageVasc,0)
        image_torch = torch.tensor(np.ascontiguousarray(image), dtype=torch.float32)

elif ResNet_type == 'input_brain_CTA_and_vasculature':
    image = np.concatenate([image,imageVasc],axis=0)
    while not len(imageVasc.shape) == 5:
        image = np.expand_dims(imageVasc,0)
        image_torch = torch.tensor(np.ascontiguousarray(image), dtype=torch.float32)


""" DeepSymNetv3 """
# Cases
DeepSymNetv3_type = 'input_only_brain_CTA' 
# input_only_brain_CTA
# input_only_vasculature
# input_brain_CTA_and_vasculature

img_cta_flip = np.flip(IMG_CTA,-3)
img_vasc_flip = np.flip(IMG_VASC,-3)

image = np.expand_dims(IMG_CTA,0)
imageFlip = np.expand_dims(img_cta_flip,0)
shape_dim1 = image.shape[1]
image = image[:,0:int(shape_dim1/2),...]
imageFlip = imageFlip[:,0:int(shape_dim1/2),...]
image = np.concatenate([image,imageFlip],axis=0) 

imageVasc = np.expand_dims(IMG_VASC,0)
imageFlipVasc = np.expand_dims(img_vasc_flip,0)
imageVasc = imageVasc[:,0:int(shape_dim1/2),...]
imageFlipVasc = imageFlipVasc[:,0:int(shape_dim1/2),...]
imageVasc = np.concatenate([imageVasc,imageFlipVasc],axis=0)


if DeepSymNetv3_type == 'input_only_brain_CTA':
    image = np.expand_dims(image,0)
    image_torch = torch.tensor(np.ascontiguousarray(image), dtype=torch.float32)


elif DeepSymNetv3_type == 'input_only_vasculature':
    imageVasc = np.expand_dims(imageVasc,0)
    image_torch = torch.tensor(np.ascontiguousarray(imageVasc), dtype=torch.float32)


elif DeepSymNetv3_type == 'input_brain_CTA_and_vasculature':
    imageVasc = np.expand_dims(imageVasc,0)
    imageFlipVasc = np.expand_dims(imageFlipVasc,0)

    imageVasc = imageVasc[:,0:int(shape_dim1/2),...]
    imageFlipVasc = imageFlipVasc[:,0:int(shape_dim1/2),...]
    imageVasc = np.concatenate([imageVasc,imageFlipVasc],axis=0)

    image = np.concatenate([image,imageVasc],axis=1)

    image_torch = torch.tensor(np.ascontiguousarray(image), dtype=torch.float32)


