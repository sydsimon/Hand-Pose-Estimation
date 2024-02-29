import numpy as np
import cv2


def xyxy2cs(x1, y1, x2, y2, img_shape, pixel_std):
    """Return center and scale of bbox in xyxy"""
    aspect_ratio = img_shape[1] * 1.0 / img_shape[0]

    center = np.zeros((2), dtype=np.float32)
    center[0] = (x1 + x2) / 2
    center[1] = (y1 + y2) / 2

    w = x2 - x1
    h = y2 - y1

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


def get_affine_transform(
    center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0
):
    """Apply affine transformation on given image"""
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_dir(src_point, rot_rad):
    """Rotate the point by `rot_rad` degree."""
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_3rd_point(a, b):
    """Return vector c that perpendicular to (a - b)."""
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def affine_transform(arr, T):
    '''
    Affine transformation of 2D points array
    Input:
        arr: (N,2)
        T: (3,3)
    Return:
        arr_: (N,2)
    '''
    assert len(arr.shape) == 2 and arr.shape[1] == 2, "Array should be in shape (N,2)"
    new_arr = np.append(arr, np.ones((arr.shape[0], 1)), axis=1).T
    new_arr = T @ new_arr
    arr_ = new_arr[:2, :].T
    return arr_


def generate_target(joints, valid_flag, image_size, heatmap_size, sigma=2):
    '''
    Input:
        joints:  (K, 2)
        valid_flag: (K,)
    Return:
        kpts_hm: (K, H_hm, W_hm) where H_hm and W_hm are heatmap dimension
        target_weight: (K,)
    '''
    num_joints = joints.shape[0]
    target_weight = valid_flag.copy()

    kpts_hm = np.zeros((num_joints, heatmap_size[0], heatmap_size[1]), dtype=np.float32)
    tmp_size = sigma * 3

    for joint_id in range(num_joints):
        if valid_flag[joint_id]:
            feat_stride = image_size / heatmap_size
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                    or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                target_weight[joint_id] = 0
                continue

            # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

            kpts_hm[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return kpts_hm, target_weight