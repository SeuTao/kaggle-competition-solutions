import numpy as np
import os
from utils import run_length_encoding
import cv2
import pandas as pd


def get_connectivity_list(mask):
    ret, labels = cv2.connectedComponents(mask, connectivity=8)
    num = np.max(labels)

    list = []
    labels = labels.astype(np.uint8)

    for i in range(1, num + 1):
        connectivity_tmp = (labels == i).astype(np.uint8) * 1
        list.append(connectivity_tmp)

    return list

def get_submission(path_list, is_minAreaRect = False):
    path = os.path.join(path_list[0])
    fea_list = os.listdir(path)
    count = 0

    output = []
    for fea_path in fea_list:
        id = fea_path.replace('.npy', '')
        output_mat=np.zeros([768, 768])

        for path in path_list:
            fea_path_tmp = os.path.join(path, fea_path)
            features = np.load(fea_path_tmp)
            output_mat += cv2.resize(features, (768, 768), cv2.INTER_LINEAR)

        output_mat = output_mat.astype(np.float32) / len(path_list)
        output_mat[output_mat >  0.5] = 1
        output_mat[output_mat <= 0.5] = 0
        output_mat=output_mat.astype(np.uint8)

        output_mat_list = get_connectivity_list(output_mat)

        # print(len(output_mat_list))

        # if is_minAreaRect:
        #     box_mask, contours, _ = cv2.findContours(output_mat, mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE)
        #
        #     for i in range(len(contours)):
        #         bounding_box = cv2.minAreaRect(contours[i])
        #         bounding_box = cv2.boxPoints(bounding_box)
        #         bounding_box = np.array(bounding_box, dtype=np.int32)
        #
        #         p0 = bounding_box[0]
        #         p1 = bounding_box[1]
        #         p2 = bounding_box[2]
        #
        #         dist1 = np.sqrt((p0[0] - p1[0]) * (p0[0] - p1[0]) + (p0[1] - p1[1]) * (p0[1] - p1[1]))
        #         dist2 = np.sqrt((p2[0] - p1[0]) * (p2[0] - p1[0]) + (p2[1] - p1[1]) * (p2[1] - p1[1]))
        #
        #         s = dist1 * dist2
        #         thres_up = 10
        #         thres_down = 10
        #
        #         if s < thres_down:
        #             print(str(s) + ' ' + str(thres_up))
        #             bounding_box = bounding_box.reshape([1, bounding_box.shape[0], bounding_box.shape[1]])
        #             cv2.fillPoly(output_mat, bounding_box, 0)
        #         elif s < thres_up:
        #             print(str(s) + ' ' + str(thres_up))
        #             bounding_box = bounding_box.reshape([1, bounding_box.shape[0], bounding_box.shape[1]])
        #             cv2.fillPoly(output_mat, bounding_box, 1)
        #
        #     if len(contours) > 5:
        #         print('!!!!!!!!!!!!!!!!!!!')
        #         cv2.imwrite('ttt.png', output_mat*255)
        #         cv2.imwrite('ttt_origin.png', origin * 255)
        #         break


        s_min = 20
        if np.sum(output_mat) == 0:

            rle_encoded = ' '.join(str(rle) for rle in run_length_encoding(np.zeros([1]).astype(np.uint8)))
            output.append([id, rle_encoded])
            # print('None')
            # print(len(output_mat_list))

        else:
            flag = False
            for output_mat_tmp in output_mat_list:
                s = np.sum(output_mat_tmp)
                if  s >= s_min:
                    rle_encoded = ' '.join(str(rle) for rle in run_length_encoding(output_mat_tmp))
                    output.append([id, rle_encoded])
                    flag = True
                    # print('!!!!!!!!!!!!!!!!!!!')
                    # cv2.imwrite('ttt.png', output_mat_tmp*255)
                    # cv2.imwrite('ttt_origin.png', output_mat_tmp * 255)
                    # break

            if  not flag:
                # output.append([id, None])
                rle_encoded = ' '.join(str(rle) for rle in run_length_encoding(np.zeros([1]).astype(np.uint8)))
                output.append([id, rle_encoded])
                print('not ')
                # else:
                #     print(s)

        count += 1
        if count % 1000 == 0:
            print(count)


    submission = pd.DataFrame(output, columns=['ImageId', 'EncodedPixels']).astype(str)
    return submission



def get_sub(empty_path, nonempty_path):
    print(len(os.listdir(nonempty_path)))
    # return

    fea_list = os.listdir(empty_path)
    count = 0

    output = []
    for fea_path in fea_list:
        id = fea_path.replace('.npy', '')
        empty_path_tmp = os.path.join(empty_path, fea_path)
        empty = np.load(empty_path_tmp)
        empty[empty > 0.5] = 1
        empty[empty <= 0.5] = 0
        empty = empty.astype(np.uint8)

        if np.sum(empty) == 0:
            rle_encoded = ' '.join(str(rle) for rle in run_length_encoding(np.zeros([1]).astype(np.uint8)))
            output.append([id, rle_encoded])
            # print('empty')
        else:

            nonempty_path_tmp = os.path.join(nonempty_path, fea_path.replace('.npy',r'.png'))
            nonempty_features = cv2.imread(nonempty_path_tmp, 0)

            # print(fea_path)
            nonempty_output_mat = cv2.resize(nonempty_features, (768, 768), cv2.INTER_LINEAR)
            nonempty_output_mat = np.asarray(nonempty_output_mat).astype(np.float32) / 255.0
            nonempty_output_mat[nonempty_output_mat > 0.5] = 1
            nonempty_output_mat[nonempty_output_mat <= 0.5] = 0

            nonempty_output_mat = nonempty_output_mat.astype(np.uint8)

            if np.sum(nonempty_output_mat) == 0:
                rle_encoded = ' '.join(str(rle) for rle in run_length_encoding(np.zeros([1]).astype(np.uint8)))
                output.append([id, rle_encoded])
                print('empty')
            else:
                output_mat_list = get_connectivity_list(nonempty_output_mat)

                for output_mat_tmp in output_mat_list:
                    rle_encoded = ' '.join(str(rle) for rle in run_length_encoding(output_mat_tmp))
                    output.append([id, rle_encoded])

        count += 1
        if count % 1000 == 0:
            print(count)


    submission = pd.DataFrame(output, columns=['ImageId', 'EncodedPixels']).astype(str)
    return submission

def get_sub_from_list(empty_path, nonempty_path_list):
    # print(len(os.listdir(nonempty_path)))
    # return

    s_min = 30
    fea_list = os.listdir(empty_path)
    count = 0

    output = []
    for fea_path in fea_list:
        id = fea_path.replace('.npy', '')
        empty_path_tmp = os.path.join(empty_path, fea_path)
        empty = np.load(empty_path_tmp)
        empty[empty > 0.5] = 1
        empty[empty <= 0.5] = 0
        empty = empty.astype(np.uint8)

        if np.sum(empty) == 0:
            rle_encoded = ' '.join(str(rle) for rle in run_length_encoding(np.zeros([1]).astype(np.uint8)))
            output.append([id, rle_encoded])
            # print('empty')
        else:
            nonempty_output_mat = np.zeros([768,768])

            for nonempty_path in nonempty_path_list:
                nonempty_path_tmp = os.path.join(nonempty_path, fea_path.replace('.npy',r'.png'))
                nonempty_features = cv2.imread(nonempty_path_tmp, 0)

                nonempty_output_mat_tmp = cv2.resize(nonempty_features, (768, 768), cv2.INTER_LINEAR)
                nonempty_output_mat_tmp = np.asarray(nonempty_output_mat_tmp).astype(np.float32) / 255.0

                nonempty_output_mat += nonempty_output_mat_tmp

            nonempty_output_mat = nonempty_output_mat / (1.0*len(nonempty_path_list))
            nonempty_output_mat[nonempty_output_mat > 0.5] = 1
            nonempty_output_mat[nonempty_output_mat <= 0.5] = 0

            nonempty_output_mat = nonempty_output_mat.astype(np.uint8)

            if np.sum(nonempty_output_mat) == 0:
                rle_encoded = ' '.join(str(rle) for rle in run_length_encoding(np.zeros([1]).astype(np.uint8)))
                output.append([id, rle_encoded])
                print('empty')
            else:
                output_mat_list = get_connectivity_list(nonempty_output_mat)


                flag = False
                for output_mat_tmp in output_mat_list:
                    s = np.sum(output_mat_tmp)
                    if s >= s_min:
                        rle_encoded = ' '.join(str(rle) for rle in run_length_encoding(output_mat_tmp))
                        output.append([id, rle_encoded])
                        flag = True
                    else:
                        print(s)
                        # print('!!!!!!!!!!!!!!!!!!!')
                        # cv2.imwrite('ttt.png', output_mat_tmp*255)
                        # cv2.imwrite('ttt_origin.png', output_mat_tmp * 255)
                        # break

                if not flag:
                    rle_encoded = ' '.join(str(rle) for rle in run_length_encoding(np.zeros([1]).astype(np.uint8)))
                    output.append([id, rle_encoded])
                    print('not ')


        count += 1
        if count % 1000 == 0:
            print(count)


    submission = pd.DataFrame(output, columns=['ImageId', 'EncodedPixels']).astype(str)
    return submission

def get_empty_dict(empty_path):
        fea_list = os.listdir(empty_path)
        # list = []

        dict = {}
        for fea_path in fea_list:
            empty_path_tmp = os.path.join(empty_path, fea_path)
            empty = np.load(empty_path_tmp)
            empty = empty.astype(np.uint8)

            if np.sum(empty) == 0:
                # list.append(fea_path)
                dict[fea_path] = 1

        print(len(dict))
        return dict






if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # path0 = r'/data/shentao/Airbus/code/models/model34_DeepSupervised_Dblock_resize768/models/fold_0/12_final_final_output'
    # # path1 = r'/data/shentao/Airbus/code/models/stnet34_V2_slim_bs32_384/models/fold_1/40_final'
    # # path2 = r'/data/shentao/Airbus/code/models/stnet34_V2_slim_bs32_384/models/fold_2/40_final'
    # # path3 = r'/data/shentao/Airbus/code/models/stnet34_V2_slim_bs32_384/models/fold_3/40_final'
    # # path4 = r'/data/shentao/Airbus/code/models/stnet34_V2_slim_bs32_384/models/fold_4/40_final'


    empty_path = r'/data/shentao/Airbus/code/models/model34_DeepSupervised_Dblock_resize768/models/fold_0/12_final_final_output'
    nonempty_path1 = r'/data/shentao/Airbus/code/models/model50_resize768_non_empty/models/fold_0/28_final'
    nonempty_path2 = r'/data/shentao/Airbus/code/models/stnet34_V2_slim_Dblock_resize768_non_empty/models/fold_0/16_final'
    # nonempty_path3 = r'/data/shentao/Airbus/code/models/model34_DeepSupervised_Dblock_resize768/models/fold_0/12_final'

    # submission = get_sub(empty_path, nonempty_path)
    submission = get_sub_from_list(empty_path, [nonempty_path1,nonempty_path2])
    submission.to_csv('510empty_model50_e28_model34_e16_resize768_non_empty_smin30.csv', index=None)