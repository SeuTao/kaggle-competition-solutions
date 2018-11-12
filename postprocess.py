import torch
import cv2
import numpy as np
import torch.nn as nn

def mask_filter(pixel_mask, link_mask, neighbors=8, scale=4):
    """
    pixel_mask: batch_size * 2 * H * W
    link_mask: batch_size * 16 * H * W
    """
    batch_size = link_mask.size(0)
    mask_height = link_mask.size(2)
    mask_width = link_mask.size(3)
    pixel_class = nn.Softmax2d()(pixel_mask)
    # print(pixel_class.shape)
    pixel_class = pixel_class[:, 1] > 0.7
    # print(pixel_class.shape)
    # pixel_class = pixel_mask[:, 1] > pixel_mask[:, 0]
    # link_neighbors = torch.ByteTensor([batch_size, neighbors, mask_height, mask_width])
    link_neighbors = torch.zeros([batch_size, neighbors, mask_height, mask_width], dtype=torch.uint8, device=pixel_mask.device)
    
    for i in range(neighbors):
        # print(link_mask[:, [2 * i, 2 * i + 1]].shape)
        tmp = nn.Softmax2d()(link_mask[:, [2 * i, 2 * i + 1]])
        # print(tmp.shape)
        link_neighbors[:, i] = tmp[:, 1] > 0.7
        # link_neighbors[:, i] = link_mask[:, 2 * i + 1] > link_mask[:, 2 * i] 
        link_neighbors[:, i] = link_neighbors[:, i] & pixel_class
    # res_mask = np.zeros([batch_size, mask_height, mask_width], dtype=np.uint8)
    pixel_class = pixel_class.cpu().numpy()
    link_neighbors = link_neighbors.cpu().numpy()
    return pixel_class, link_neighbors

def mask_to_box(pixel_mask, link_mask, neighbors=8, scale=4):
    """
    pixel_mask: batch_size * 2 * H * W
    link_mask: batch_size * 16 * H * W
    """
    def distance(a, b):
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
    def short_side_filter(bounding_box):
        for i, point in enumerate(bounding_box):
            if distance(point, bounding_box[(i+1)%4]) < 5**2:
                return True # ignore it
        return False # do not ignore
    batch_size = link_mask.size(0)
    mask_height = link_mask.size(2)
    mask_width = link_mask.size(3)
    pixel_class = nn.Softmax2d()(pixel_mask)
    # print(pixel_class.shape)
    pixel_class = pixel_class[:, 1] > 0.7
    # pixel_class = pixel_mask[:, 1] > pixel_mask[:, 0]
    # link_neighbors = torch.ByteTensor([batch_size, neighbors, mask_height, mask_width])
    link_neighbors = torch.zeros([batch_size, neighbors, mask_height, mask_width], dtype=torch.uint8, device=pixel_mask.device)
    
    for i in range(neighbors):
        # print(link_mask[:, [2 * i, 2 * i + 1]].shape)
        tmp = nn.Softmax2d()(link_mask[:, [2 * i, 2 * i + 1]])
        # print(tmp.shape)
        link_neighbors[:, i] = tmp[:, 1] > 0.7
        # link_neighbors[:, i] = link_mask[:, 2 * i + 1] > link_mask[:, 2 * i] 
        link_neighbors[:, i] = link_neighbors[:, i] & pixel_class
    # res_mask = np.zeros([batch_size, mask_height, mask_width], dtype=np.uint8)
    all_boxes = []
    # res_masks = []
    for i in range(batch_size):
        res_mask = func(pixel_class[i], link_neighbors[i])
        box_num = np.amax(res_mask)
        # print(res_mask.any())
        bounding_boxes = []
        for i in range(1, box_num + 1):
            box_mask = (res_mask == i).astype(np.uint8)
            # res_masks.append(box_mask)
            if box_mask.sum() < 100:
                pass
                # print("<150")
                # continue
            box_mask, contours, _ = cv2.findContours(box_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
            # print(contours[0])
            bounding_box = cv2.minAreaRect(contours[0])
            bounding_box = cv2.boxPoints(bounding_box)
            if short_side_filter(bounding_box):
                # print("<5")
                pass
                continue
            # bounding_box = bounding_box.reshape(8)
            bounding_box = np.clip(bounding_box * scale, 0, 128 * scale - 1).astype(np.int)
            # import IPython
            # IPython.embed()
            bounding_boxes.append(bounding_box)
        all_boxes.append(bounding_boxes)
    return all_boxes

def get_neighbors(h_index, w_index):
    res = []
    res.append((h_index - 1, w_index - 1))
    res.append((h_index - 1, w_index))
    res.append((h_index - 1, w_index + 1))
    res.append((h_index, w_index + 1))
    res.append((h_index + 1, w_index + 1))
    res.append((h_index + 1, w_index))
    res.append((h_index + 1, w_index - 1))
    res.append((h_index, w_index - 1))
    return res

def func(pixel_cls, link_cls):
    def joint(pointa, pointb):
        roota = find_root(pointa)
        rootb = find_root(pointb)
        if roota != rootb:
            group_mask[rootb] = roota
            # group_mask[pointb] = roota
            # group_mask[pointa] = roota
        return

    def find_root(pointa):
        root = pointa
        while group_mask.get(root) != -1:
            root = group_mask.get(root)
        return root

    pixel_cls = pixel_cls.cpu().numpy()
    link_cls = link_cls.cpu().numpy()

    # import IPython
    # IPython.embed()

    # print(pixel_cls.any())
    # print(np.where(pixel_cls))
    pixel_points = list(zip(*np.where(pixel_cls)))
    h, w = pixel_cls.shape
    group_mask = dict.fromkeys(pixel_points, -1)
    # print(group_mask)

    for point in pixel_points:
        h_index, w_index = point
        # print(point)
        neighbors = get_neighbors(h_index, w_index)
        for i, neighbor in enumerate(neighbors):
            nh_index, nw_index = neighbor
            if nh_index < 0 or nw_index < 0 or nh_index >= h or nw_index >= w:
                continue
            if pixel_cls[nh_index, nw_index] == 1 and link_cls[i, h_index, w_index] == 1:
                joint(point, neighbor)

    res = np.zeros(pixel_cls.shape, dtype=np.uint8)
    root_map = {}
    for point in pixel_points:
        h_index, w_index = point
        root = find_root(point)
        if root not in root_map:
            root_map[root] = len(root_map) + 1
        res[h_index, w_index] = root_map[root]

    return res


if __name__ == '__main__':
    # file_path = r'/data/shentao/Airbus/code/models/model34_DeepSupervised_resize384/samples/37_1500_output.png'
    # img = cv2.imread(file_path, 0)
    # print(img.shape)
    #
    # for i in range(30,100):
    #     # i = 14
    #     img_tmp = img[i*384:(i+1)*384, 384:]
    #     print(img_tmp.shape)
    #
    #     cv2.imwrite('tmp.png', img_tmp)
    #
    #     file_path = r'tmp.png'
    #     img_tmp = cv2.imread(file_path, 0)
    #     box_mask, contours, _ = cv2.findContours(img_tmp, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    #     print(len(contours))
    #
    #     if len(contours) > 3:
    #         break

    file_path = r'tmp.png'
    img_tmp = cv2.imread(file_path, 0)
    box_mask, contours, _ = cv2.findContours(img_tmp, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    print(len(contours))

    for i in range(len(contours)):
        bounding_box = cv2.minAreaRect(contours[i])
        bounding_box = cv2.boxPoints(bounding_box)

        bounding_box = np.array(bounding_box, dtype=np.int32)
        bounding_box = bounding_box.reshape([1,bounding_box.shape[0],bounding_box.shape[1]])
        print(bounding_box.shape)

        cv2.fillPoly(img_tmp, bounding_box, 255)

        # print(bounding_box)
        # cv2.line(img_tmp, (int(bounding_box[0][0]),int(bounding_box[0][1])), (int(bounding_box[1][0]),int(bounding_box[1][1])), 255, 2)
        # cv2.line(img_tmp, (int(bounding_box[1][0]), int(bounding_box[1][1])),(int(bounding_box[2][0]), int(bounding_box[2][1])), 255, 2)
        # cv2.line(img_tmp, (int(bounding_box[2][0]), int(bounding_box[2][1])),(int(bounding_box[3][0]), int(bounding_box[3][1])), 255, 2)
        # cv2.line(img_tmp, (int(bounding_box[3][0]), int(bounding_box[3][1])),(int(bounding_box[0][0]), int(bounding_box[0][1])), 255, 2)

    cv2.imwrite('tmp_bbox.png', img_tmp)

    # cv2.floodFill(img_tmp, img_tmp, (0, 0), 255)
    # cv2.imwrite('tmp_bbox_fill.png', img_tmp)

    a = np.array([[[10, 10], [100, 10], [100, 100], [10, 100]]], dtype=np.int32)
    print(a.shape)








