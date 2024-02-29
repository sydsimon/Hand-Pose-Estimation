import matplotlib.pyplot as plt
import torch
import numpy as np


def vis_data(img, kpts, title=None):
    """
    Given image and kpts, plot hand joints with each finger
    """
    single_hand_index = np.array([[0,1,2,3,4],
                             [0,5,6,7,8],
                             [0,9,10,11,12],
                             [0,13,14,15,16],
                             [0,17,18,19,20]])
    color_dict = {0:'tab:blue', 1:'tab:orange', 2:'tab:green', 3:'tab:red', 4:'tab:purple'}

    plt.figure(figsize=(5,5))
    plt.imshow(img)
    for i, finger_index in enumerate(single_hand_index):
        curr_finger_kpts = kpts[finger_index]
        plt.plot(curr_finger_kpts[:,0], curr_finger_kpts[:,1], marker='o', markersize=2, color=color_dict[i])
    if title:
        plt.title(title)

def inverse_normalize(tensor, mean, std):
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    tensor = torch.clip(tensor, 0, 1)
    return tensor