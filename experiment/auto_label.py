# -*-coding:utf-8-*-
import SimpleITK as sitk
import numpy as np
import cv2, os
import bottleneck
import matplotlib.pyplot as plt
import scipy.cluster.vq as vq

def get_binary_mask(image, threshold=255):
    assert image.dtype == 'uint8', "input image must has data type 'uint8'"
    mask = image.copy()
    # label chromosome region as 1
    mask[image < threshold] = 1
    # label background as 0
    mask[image >= threshold] = 0
    return mask

def check_binary(binary_mask):
    num_zero = np.sum(binary_mask == 0)
    num_one = np.sum(binary_mask == 1)
    return num_zero + num_one == np.prod(binary_mask.shape)

def get_connected_component_sitk(array, if_binary=True, threshold=0):
    '''

    '''
    if if_binary:
        assert check_binary(array), 'input array must be binary'

    sitk_image = sitk.GetImageFromArray(array)

    label = sitk.ConnectedComponent(sitk_image > threshold)
    stat = sitk.LabelIntensityStatisticsImageFilter()
    stat.Execute(label, sitk_image)
    #print sitk.GetArrayFromImage(sitk.LabelOverlay(sitk_image, label)).shape
    plt.imshow(sitk.GetArrayFromImage(sitk.LabelOverlay(sitk_image, label)))
    plt.show()

def get_connected_component_cv(array, if_binary=True, threshold=200):

    if if_binary:
        assert check_binary(array), 'input array must be binary'

    ret, label = cv2.connectedComponents(array)
    #print np.max(label)
    num_component = np.max(label)
    chromosome_num = 0
    for i in range(num_component):
        #print 'the %s-th connect component has size: %s' %(i+1, np.sum((label == i+1)))
        if np.sum((label == i+1)) < threshold:
            label[label == i+1] = 0
        else:
            chromosome_num += 1
            label[label == i+1] = chromosome_num
    return label, chromosome_num

def get_com(label, chromosome_num):
    # for each connected component(chromosome), get the x, y coordinates of the center of mass
    coord_list = np.zeros((chromosome_num, 2))
    for i in range(chromosome_num):
        coord_grp = np.where(label == i+1)
        coord_list[i, 0] = np.mean(coord_grp[0])
        coord_list[i, 1] = np.mean(coord_grp[1])
    return coord_list

def draw_mask(label, img_num):
    # Map component labels to hue val
    label_hue = np.uint8(255 * label / np.max(label))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to white
    labeled_img[label_hue == 0] = 255

    #cv2.imshow('labeled.png', labeled_img)
    #cv2.imwrite('label_%d.png' %(img_num), labeled_img)
    return labeled_img

def draw_chromosome(label, chromosome_num):
    for i in range(chromosome_num):
        label_chromosome = label.copy()
        label_chromosome[label != i+1] = 0
        draw_mask(label_chromosome, i+1)

def get_euclidean_distance(coord_1, coord_2):
    assert len(coord_1) == 2 and len(coord_2) == 2, 'input must be 2-tuple'
    return np.sqrt((coord_1[0] - coord_2[0]) ** 2 + (coord_1[1] - coord_2[1]) ** 2)


def get_distance_matrix(coord_list):
    distance_matrix = np.zeros((coord_list.shape[0], coord_list.shape[0]))
    for i in range(coord_list.shape[0]):
        for j in range(coord_list.shape[0]):
            if i == j:
                distance_matrix[i, j] = 9999
            else:
                distance_matrix[i, j] = get_euclidean_distance(coord_list[i], coord_list[j])
    return distance_matrix

def least_distance_match(distance_matrix):
    matched_index_list = []
    for i in range(distance_matrix.shape[0]):
        #print distance_matrix[i]
        matched_index_list.append(np.argmin(distance_matrix[i]))
    #print matched_index_list
    return matched_index_list

def check_paired(matched_index_list):
    for i in range(len(matched_index_list)):
        assert matched_index_list[matched_index_list[i]] == i, 'COMs are not paired'
    return True

def get_paired_coord_list(matched_index_list, coord_list):
    '''

    :param matched_index_list: keep track of the unpaired indices for each paired group
    :param coord_list:
    :return:
    '''
    assert len(coord_list) % 2 == 0,' coord_list must contain even number of points'
    unpair_to_pair_map = [[0, 0] for _ in range(len(coord_list)/2)]
    matched_list = [False for _ in range(len(coord_list))]
    paired_coord_list = []
    if check_paired(matched_index_list):
        for i in range(len(coord_list)):
            if not matched_list[i]:
                coord_1 = coord_list[i]
                coord_2 = coord_list[matched_index_list[i]]
                unpair_to_pair_map[len(paired_coord_list)] = [i, matched_index_list[i]]
                paired_coord_list.append((coord_1 + coord_2)/2)
                matched_list[i] = True
                matched_list[matched_index_list[i]] = True
    return unpair_to_pair_map, matched_list, paired_coord_list

def get_paired_mask(unpair_to_pair_map, label, has_Y=False):
    new_label = np.zeros(label.shape)
    for i in range(len(unpair_to_pair_map)):
        new_label[label == unpair_to_pair_map[i][0] + 1] = i+1
        new_label[label == unpair_to_pair_map[i][1] + 1] = i+1
        if has_Y and i == 22:
            new_label[label == unpair_to_pair_map[22][0] + 1] = 23
            new_label[label == unpair_to_pair_map[22][1] + 1] = 24
            return new_label
    return new_label

def get_sorted_paired_idx(paired_coord_list, num_per_group=[5, 7, 6, 5], iteration=0, max_iteration=3, sorted_paired_idx=[]):
    if iteration <= max_iteration:
        coord_list_x = [i[0] for i in paired_coord_list]
        coord_list_y = [i[1] for i in paired_coord_list]
        first_group_idx = bottleneck.argpartition(coord_list_x, num_per_group[iteration])[:num_per_group[iteration]]
        first_group_y = [coord_list_y[i] for i in first_group_idx]
        tmp_sorted_first_group_idx = sorted(range(len(first_group_y)), key=lambda k: first_group_y[k])
        sorted_first_group_idx = [first_group_idx[i] for i in tmp_sorted_first_group_idx]
        new_sorted_paired_idx = sorted_paired_idx
        new_sorted_paired_idx += sorted_first_group_idx
        new_paired_coord_list = paired_coord_list
        for i in first_group_idx:
            new_paired_coord_list[i] = [9999, 9999]

        return get_sorted_paired_idx(new_paired_coord_list, num_per_group=num_per_group, iteration=iteration+1,
                                      sorted_paired_idx=new_sorted_paired_idx)
    else:
        return sorted_paired_idx

def get_extreme_num(label, region_num):
    '''
    :param label: label mask
    :param region_num: number of regions
    :return: a slit of length region num, each element looks like [xmin, xmax, ymin, ymax]
    '''
    extreme_num_list = []
    for i in range(region_num):
        xmin = np.min(np.where(label == i + 1)[0])
        xmax = np.max(np.where(label == i + 1)[0])
        ymin = np.min(np.where(label == i + 1)[1])
        ymax = np.max(np.where(label == i + 1)[1])
        extreme_num_list.append([xmin, xmax, ymin, ymax])
    return extreme_num_list

def draw_label(label, region_num, img, img_name='', has_Y=False):
    img = img.copy()
    extreme_num_list = get_extreme_num(label, region_num)
    #print extreme_num_list
    for i in range(region_num):
        xmin = extreme_num_list[i][0]
        xmax = extreme_num_list[i][1]
        ymin = extreme_num_list[i][2]
        ymax = extreme_num_list[i][3]
        cv2.line(img, (ymin - 10, xmax + 10), (ymin - 10, xmax + 20), color=(0, 0, 0), thickness=1)
        cv2.line(img, (ymin - 10, xmax + 20), (ymax + 10, xmax + 20), color=(0, 0, 0), thickness=1)
        cv2.line(img, (ymax + 10, xmax + 20), (ymax + 10, xmax + 10), color=(0, 0, 0), thickness=1)
        if i <= 21:
            cv2.putText(img, '%d' %(i + 1), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, org=((ymin + ymax)/2 - 10, xmax + 50),
                        color=(0, 0, 0))
            continue
        if not has_Y:
            if i == 23:
                cv2.putText(img, 'Y', fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, org=((ymin + ymax) / 2 - 10, xmax + 50),
                            color=(0, 0, 0))
            else:
                cv2.putText(img, 'X', fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, org=((ymin + ymax)/2 - 10, xmax + 50),
                            color=(0, 0, 0))
        else:
            cv2.putText(img, 'Y', fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, org=((ymin + ymax) / 2 - 10, xmax + 50),
                        color=(0, 0, 0))

    cv2.imwrite('%s_labeled.png' %(img_name), img)


if __name__ == '__main__':
    root_dir = '/media/tx-eva-cc/data/chromosome/2018_11_16/2018_11_16/sorted'
    for img_name in os.listdir(root_dir):
        print img_name
        img_dir = os.path.join(root_dir, img_name)
        image = cv2.imread(img_dir)
        # convert RGB image to grayscale
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = get_binary_mask(gray_img)
        label, chromosome_num = get_connected_component_cv(mask)
        #draw_chromosome(label, chromosome_num)
        coord_list = get_com(label, chromosome_num)
        #print coord_list.shape[0]
        #print vq.kmeans2(coord_list, k=23, iter=1000)
        distance_matrix = get_distance_matrix(coord_list)
        matched_index_list = least_distance_match(distance_matrix)
        check_paired(matched_index_list)
        unpair_to_pair_map, matched_list, paired_coord_list = get_paired_coord_list(matched_index_list, coord_list)
        #print unpair_to_pair_map, matched_list, paired_coord_list
        sorted_paired_idx = get_sorted_paired_idx(paired_coord_list,
                                                  num_per_group=[5, 7, 6, 5], iteration=0, max_iteration=3, sorted_paired_idx=[])
        sorted_unpair_to_pair_map = [unpair_to_pair_map[i] for i in sorted_paired_idx]
        # check whether there is Y chromosome in the image
        if distance_matrix[sorted_unpair_to_pair_map[22][0]][sorted_unpair_to_pair_map[22][1]] > 100:
            has_Y = True
            region_num = 24
        else:
            has_Y = False
            region_num = 23
        new_label = get_paired_mask(sorted_unpair_to_pair_map, label, has_Y=has_Y)
        #draw_chromosome(new_label, chromosome_num/2)
        print len(sorted_unpair_to_pair_map)
        draw_img = draw_mask(new_label, 0)
        draw_label(new_label, region_num, draw_img, img_name=img_name)
