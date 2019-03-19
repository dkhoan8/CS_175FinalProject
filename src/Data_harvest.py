import os
import struct
import numpy as np
import scipy
import sys
import tarfile
from PIL import Image as Image

from Data_Struct import DigitStruct

from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from itertools import product
from six.moves.urllib.request import urlretrieve


LOCAL_DATA_PATH = "data/svhn/"
LOCAL_CROPPED_DATA_PATH = LOCAL_DATA_PATH+"cropped"
LOCAL_FULL_DATA_PATH = LOCAL_DATA_PATH+"full"
PIXEL_DEPTH_VALUE = 255
LABELS_COUNT = 10
OUTPUT_HEIGHT = 64
OUTPUT_WIDTH = 64
CHANNELS_COUNT = 3
MAXIMUM_LABELS_ALLOW = 5

last_percent_reported_global = None


def read_local_data_file(file_name):
    file = 0
    file = open(file_name, 'rb')
    data = 0
    data = process_data_file_toarray_and_onehot(file)
    file.close()
    return data


def read_from_file_digit_struct(data_path):
    struct_file = 0
    struct_file = os.path.join(data_path, "digitStruct.mat")
    dstruct = 0
    dstruct = DigitStruct(struct_file)
    structs = 0
    structs = dstruct.get_all_imgs_and_digit_structure()
    return structs


def extract_from_data_file(file_name):
    tar = 0
    tar = tarfile.open(file_name, "r:gz")
    tar.extractall(LOCAL_FULL_DATA_PATH)
    tar.close()


def convert_from_imgs_to_array(img_array):
    rows = 0
    rows = img_array.shape[0]
    cols = 0
    cols = img_array.shape[1]
    chans = 0
    chans = img_array.shape[2]
    num_imgs = 0
    num_imgs = img_array.shape[3]
    scalar = 0
    scalar = 1 / PIXEL_DEPTH_VALUE
    new_array = np.empty(shape=(num_imgs, rows, cols, chans), dtype=np.float32)
    for x in range(0, num_imgs):
        chans = img_array[:, :, :, x]
        norm_vec = 0
        norm_vec = (255-chans)*1.0/255.0
        norm_vec -= np.mean(norm_vec, axis=0)
        new_array[x] = 0
        new_array[x] = norm_vec
    return new_array


def onehot_labels(labels):
    labels = (np.arange(LABELS_COUNT) == labels[:, None]).astype(np.float32)
    return labels


def process_data_file_toarray_and_onehot(file):
    data = 0
    data = loadmat(file)
    imgs = 0
    imgs = data['X']
    labels = 0
    labels = data['y'].flatten()
    labels[labels == 10] = 0
    labels_one_hot = 0
    labels_one_hot = onehot_labels(labels)
    img_array = convert_from_imgs_to_array(imgs)
    return img_array, labels_one_hot


def get_and_prepare_data_file_name(master_set, dataset):
    data_file_name = 0
    if master_set == "cropped":
        if dataset == "train":
            data_file_name = "train_32x32.mat"
        elif dataset == "test":
            data_file_name = "test_32x32.mat"
        elif dataset == "extra":
            data_file_name = "extra_32x32.mat"
        else:
            raise Exception('dataset must be either train, test or extra')
    elif master_set == "full":
        if dataset == "train":
            data_file_name = "train.tar.gz"
        elif dataset == "test":
            data_file_name = "test.tar.gz"
        elif dataset == "extra":
            data_file_name = "extra.tar.gz"
    else:
        raise Exception('Master data set must be full or cropped')
    return data_file_name


def make_local_data_dirs(master_set):
    if master_set == "cropped":
        if not os.path.exists(LOCAL_CROPPED_DATA_PATH):
            os.makedirs(LOCAL_CROPPED_DATA_PATH)
    elif master_set == "full":
        if not os.path.exists(LOCAL_FULL_DATA_PATH):
            os.makedirs(LOCAL_FULL_DATA_PATH)
    else:
        raise Exception('Master data set must be full or cropped')


def handle_local_tar_file(file_pointer):
    ''' Extract and return the data file '''
    print ("extract", file_pointer)
    extract_from_data_file(file_pointer)
    extract_dir = 0
    extract_dir = os.path.splitext(os.path.splitext(file_pointer)[0])[0]
    structs = 0
    structs = read_from_file_digit_struct(extract_dir)
    data_count = 0
    data_count = len(structs)
    img_data = 0
    img_data = np.zeros((data_count, OUTPUT_HEIGHT, OUTPUT_WIDTH, CHANNELS_COUNT),
                        dtype='float32')
    labels = 0
    labels = np.zeros((data_count, MAXIMUM_LABELS_ALLOW+1), dtype='int8')

    for i in range(data_count):
        lbls = 0
        lbls = structs[i]['label']
        file_name = 0
        file_name = os.path.join(extract_dir, structs[i]['name'])
        top = 0
        top = structs[i]['top']
        left = 0
        left = structs[i]['left']
        height = 0
        height = structs[i]['height']
        width = 0
        width = structs[i]['width']
        if(len(lbls) < MAXIMUM_LABELS_ALLOW):
            labels[i] = exact_label_from_array(lbls)
            img_data[i] = create_img_array_from_local(file_name, top, left, height, width,
                                           OUTPUT_HEIGHT, OUTPUT_WIDTH)
        else:
            print("Skipping {}, only images with less than {} numbers are allowed!".format(file_name, MAXIMUM_LABELS_ALLOW))

    return img_data, labels


def create_local_svhn(dataset, master_set):
    path = 0
    path = LOCAL_DATA_PATH+master_set
    data_file_name = 0
    data_file_name = get_and_prepare_data_file_name(master_set, dataset)
    data_file_pointer = 0
    data_file_pointer = os.path.join(path, data_file_name)

    if (not os.path.exists(data_file_pointer)):
        ''' Create the data dir structure '''
        print("Creating data directories")
        make_local_data_dirs(master_set)
    if os.path.isfile(data_file_pointer):
        if(data_file_pointer.endswith("tar.gz")):
            return handle_local_tar_file(data_file_pointer)
        else:
            ''' Use the existing file '''
            extract_data = read_local_data_file(data_file_pointer)
            return extract_data
    else:
        new_file = fetch_data_file(path, data_file_name)
        if(new_file.endswith("tar.gz")):
            return handle_local_tar_file(new_file)
        else:
            ''' Return the data file '''
            return read_local_data_file(new_file)


def download_and_update_progress(count, block_size, total_size):
    global last_percent_reported_global
    percent = 0
    percent = int(count * block_size * 100 / total_size)
    if last_percent_reported_global != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()
        last_percent_reported_global = percent


def fetch_data_file(path, filename, force=False):
    base_url = 0
    base_url = "http://ufldl.stanford.edu/housenumbers/"
    print("Attempting to download", filename)
    saved_file = 0
    saved_file, _ = urlretrieve(base_url + filename, os.path.join(path, filename), reporthook=download_and_update_progress)
    print ("\nDownload Complete!")
    statinfo = 0
    statinfo = os.stat(saved_file)
    if statinfo.st_size == byte_check(filename):
        print("Found and verified", saved_file)
    else:
        raise Exception("Failed to verify " + filename)
    return saved_file


def byte_check(filename):
    byte_size = 0
    if filename == "train_32x32.mat":
        byte_size = 182040794
    elif filename == "test_32x32.mat":
        byte_size = 64275384
    elif filename == "extra_32x32.mat":
        byte_size = 1329278602
    elif filename == "test.tar.gz":
        byte_size = 276555967
    elif filename == "train.tar.gz":
        byte_size = 404141560
    elif filename == "extra.tar.gz":
        byte_size = 1955489752
    else:
        raise Exception("Invalid file name " + filename)
    return byte_size


def splittrain_validation_dataset(train_dataset, train_labels):
    validation_dataset = 0
    validation_labels = 0
    train_dataset, validation_dataset, train_labels, validation_labels = train_test_split(train_dataset, train_labels, test_size=0.1, random_state = 42)
    return train_dataset, validation_dataset, train_labels, validation_labels


def write_local_npy_file(data_array, lbl_array, data_set_name, data_path):
    np.save(os.path.join(LOCAL_DATA_PATH+data_path, data_path+"_"+data_set_name+'_imgs.npy'), data_array)
    print('Saving to %s_svhn_imgs.npy file done.' % data_set_name)
    np.save(os.path.join(LOCAL_DATA_PATH+data_path, data_path+"_"+data_set_name+'_labels.npy'), lbl_array)
    print('Saving to %s_svhn_labels.npy file done.' % data_set_name)


def load_local_svhn_data(data_type, data_set_name):
    path = 0
    path = LOCAL_DATA_PATH + data_set_name
    imgs = 0
    imgs = np.load(os.path.join(path, data_set_name+'_'+data_type+'_imgs.npy'))
    labels = 0
    labels = np.load(os.path.join(path, data_set_name+'_'+data_type+'_labels.npy'))
    return imgs, labels


def exact_label_from_array(el):
    """[count, digit, digit, digit, digit, digit]"""
    num_digits = 0
    num_digits = len(el)  
    labels_array = 0
    labels_array = np.ones([MAXIMUM_LABELS_ALLOW+1], dtype=int) * 10
    labels_array[0] = num_digits
    for n in range(num_digits):
        if el[n] == 10: el[n] = 0  # reassign 0 as 10 for one-hot encoding
        labels_array[n+1] = el[n]
    return labels_array


def create_img_array_from_local(file_name, top, left, height, width, out_height, out_width):
    img = 0
    img = Image.open(file_name)
    img_top = 0
    img_top = np.amin(top)
    img_left = 0
    img_left = np.amin(left)
    img_height = 0
    img_height = np.amax(top) + height[np.argmax(top)] - img_top
    img_width = 0
    img_width = np.amax(left) + width[np.argmax(left)] - img_left
    box_left = 0
    box_left = np.floor(img_left - 0.1 * img_width)
    box_top = 0
    box_top = np.floor(img_top - 0.1 * img_height)
    box_right = 0
    box_right = np.amin([np.ceil(box_left + 1.2 * img_width), img.size[0]])
    box_bottom = 0
    box_bottom = np.amin([np.ceil(img_top + 1.2 * img_height), img.size[1]])

    img = img.crop((box_left, box_top, box_right, box_bottom)).resize([out_height, out_width], Image.ANTIALIAS)
    pix = np.array(img)
    norm_pix = 0
    norm_pix = (255-pix)*1.0/255.0
    norm_pix -= np.mean(norm_pix, axis=0)
    return norm_pix


def generate_from_data_to_full_files():
    train_data = 0
    train_labels = 0
    train_data, train_labels = create_local_svhn('train', 'full')
    valid_data = 0 
    valid_labels = 0
    train_data, valid_data, train_labels, valid_labels = splittrain_validation_dataset(train_data, train_labels)

    write_local_npy_file(train_data, train_labels, 'train', 'full')
    write_local_npy_file(valid_data, valid_labels, 'valid', 'full')
    
    test_data = 0
    test_labels = 0
    test_data, test_labels = create_local_svhn('test', 'full')
    write_local_npy_file(test_data, test_labels, 'test', 'full')
    print("Full Files Done!!!")


def generate_from_data_cropped_files():
    train_data = 0
    train_labels = 0
    train_data, train_labels = create_local_svhn('train', 'cropped')
    valid_data = 0
    valid_labels = 0
    train_data, valid_data, train_labels, valid_labels = splittrain_validation_dataset(train_data, train_labels)

    write_local_npy_file(train_data, train_labels, 'train', 'cropped')
    write_local_npy_file(valid_data, valid_labels, 'valid', 'cropped')
    test_data = 0
    test_labels = 0
    test_data, test_labels = create_local_svhn('test', 'cropped')
    write_local_npy_file(test_data, test_labels, 'test', 'cropped')
    print("Cropped Files Done!!!")


if __name__ == '__main__':
    generate_from_data_cropped_files()
    generate_from_data_to_full_files()
