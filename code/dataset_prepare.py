import pandas as pd
from PIL import Image


def get_data_df_drive(shuffle=False):
    ROOT_DIR = './data/DRIVE'
    all_size = 40
    train_size = 32
    valid_size = 8

    paths = []
    classes = []
    for i in range(0, all_size):
        j = i + 1
        if j < 10:
            path = f'{ROOT_DIR}/images/0{j}_test.tif'
            class_path = f'{ROOT_DIR}/manual/0{j}_manual1.gif'
        elif j < 21:
            path = f'{ROOT_DIR}/images/{j}_test.tif'
            class_path = f'{ROOT_DIR}/manual/{j}_manual1.gif'
        else:
            path = f'{ROOT_DIR}/images/{j}_training.tif'
            class_path = f'{ROOT_DIR}/manual/{j}_manual1.gif'
        x = Image.open(path)
        try:
            x = x.convert('RGB') # To deal with some grayscale images in the data
        except:
            pass
        paths.append(x)
        y = Image.open(class_path)
        try:
            y = y.convert('RGB') # To deal with some grayscale images in the data
        except:
            pass
        classes.append(y)
    datas = {
        'path': paths,
        'class': classes,
    }
    data_df = pd.DataFrame(datas, columns=['path', 'class'])
    if shuffle:
        data_df = data_df.sample(frac=1).reset_index(drop=True) # Shuffles the data
    return data_df, all_size, train_size, valid_size


def get_data_df_stare(shuffle=False):
    ROOT_DIR = './data/STARE'

    ids = ['001', '002', '003', '004', '005', '044', '077', '081', '082', '139', '162', '163', '235', '236', '239', '240', '255', '291', '319', '324']
    paths = []
    classes = []
    all_size = len(ids)
    valid_size = int(all_size * 0.2001)
    train_size = all_size - valid_size
    for one_id in ids:
        path = f'{ROOT_DIR}/stare-images/im0{one_id}.ppm'
        class_path = f'{ROOT_DIR}/labels-ah/im0{one_id}.ah.pgm'
        x = Image.open(path)
        try:
            x = x.convert('RGB') # To deal with some grayscale images in the data
        except:
            print('something x')
            pass
        paths.append(x)
        #print(x)
        y = Image.open(class_path)
        try:
            y = y.convert('RGB') # To deal with some grayscale images in the data
        except:
            print('something y')
            pass
        classes.append(y)
        #print(y)
    datas = {
        'path': paths,
        'class': classes,
    }
    data_df = pd.DataFrame(datas, columns=['path', 'class'])
    if shuffle:
        data_df = data_df.sample(frac=1).reset_index(drop=True) # Shuffles the data
    return data_df, all_size, train_size, valid_size


def get_data_df_av_wide(shuffle=False):
    ROOT_DIR = './data/AV-WIDE'

    paths = []
    classes = []
    all_size = 30
    valid_size = int(all_size * 0.2001)
    train_size = all_size - valid_size
    for i in range(30):
        path = f'{ROOT_DIR}/images/wide_image_{i+1:02d}.png'
        class_path = f'{ROOT_DIR}/manual/wide_image_{i+1:02d}_vessels.png'
        x = Image.open(path)
        try:
            x = x.convert('RGB') # To deal with some grayscale images in the data
        except:
            print('something x')
            pass
        paths.append(x)
        #print(x)
        y = Image.open(class_path)
        try:
            y = y.convert('RGB') # To deal with some grayscale images in the data
        except:
            print('something y')
            pass
        classes.append(y)
        #print(y)
    datas = {
        'path': paths,
        'class': classes,
    }
    data_df = pd.DataFrame(datas, columns=['path', 'class'])
    if shuffle:
        data_df = data_df.sample(frac=1).reset_index(drop=True) # Shuffles the data
    return data_df, all_size, train_size, valid_size


def get_data_df_chasedb(shuffle=False):
    ROOT_DIR = './data/CHASEDB'

    paths = []
    classes = []
    all_size = 14*2
    valid_size = 6
    train_size = all_size - valid_size
    for i in range(14):
        for j in ['L', 'R']:
            path = f'{ROOT_DIR}/images/Image_{i+1:02d}{j}.jpg'
            class_path = f'{ROOT_DIR}/manual/Image_{i+1:02d}{j}_1stHO.png'
            x = Image.open(path)
            try:
                x = x.convert('RGB') # To deal with some grayscale images in the data
            except:
                print('something x')
                pass
            paths.append(x)
            #print(x)
            y = Image.open(class_path)
            try:
                y = y.convert('RGB') # To deal with some grayscale images in the data
            except:
                print('something y')
                pass
            classes.append(y)
        #print(y)
    datas = {
        'path': paths,
        'class': classes,
    }
    data_df = pd.DataFrame(datas, columns=['path', 'class'])
    if shuffle:
        data_df = data_df.sample(frac=1).reset_index(drop=True) # Shuffles the data
    return data_df, all_size, train_size, valid_size


def get_data_df_vevio_frame(shuffle=False):
    ROOT_DIR = './data/VEVIO'

    paths = []
    classes = []
    ids = [2,3,4,6,7,8,10,11,12,13,14,15,17,20,22,23]
    all_size = 16
    valid_size = 3
    train_size = all_size - valid_size
    for id in ids:
        path = f'{ROOT_DIR}/frames/neo{id:02d}od1_d.png'
        class_path = f'{ROOT_DIR}/frames_manual_01_bw/bw_neo{id:02d}od1_d_black.png'
        x = Image.open(path)
        try:
            x = x.convert('RGB') # To deal with some grayscale images in the data
        except:
            print('something x')
            pass
        paths.append(x)
        #print(x)
        y = Image.open(class_path)
        try:
            y = y.convert('RGB') # To deal with some grayscale images in the data
        except:
            print('something y')
            pass
        classes.append(y)
        #print(y)
    datas = {
        'path': paths,
        'class': classes,
    }
    data_df = pd.DataFrame(datas, columns=['path', 'class'])
    if shuffle:
        data_df = data_df.sample(frac=1).reset_index(drop=True) # Shuffles the data
    return data_df, all_size, train_size, valid_size


def get_data_df_vevio_mosaics(shuffle=False):
    ROOT_DIR = './data/VEVIO'

    paths = []
    classes = []
    ids = [2,3,4,6,7,8,10,11,12,13,14,15,17,20,22,23]
    all_size = 16
    valid_size = 3
    train_size = all_size - valid_size
    for id in ids:
        path = f'{ROOT_DIR}/mosaics/neo{id:02d}od1_m.png'
        class_path = f'{ROOT_DIR}/mosaics_manual_01_bw/bw_neo{id:02d}od1_m_black.png'
        x = Image.open(path)
        try:
            x = x.convert('RGB') # To deal with some grayscale images in the data
        except:
            print('something x')
            pass
        paths.append(x)
        #print(x)
        y = Image.open(class_path)
        try:
            y = y.convert('RGB') # To deal with some grayscale images in the data
        except:
            print('something y')
            pass
        classes.append(y)
        #print(y)
    datas = {
        'path': paths,
        'class': classes,
    }
    data_df = pd.DataFrame(datas, columns=['path', 'class'])
    if shuffle:
        data_df = data_df.sample(frac=1).reset_index(drop=True) # Shuffles the data
    return data_df, all_size, train_size, valid_size


def get_data_frame(dataset_name):
    if dataset_name == 'DRIVE':
        data_df, all_size, train_size, valid_size = get_data_df_drive(shuffle=True)
    elif dataset_name == 'STARE':
        data_df, all_size, train_size, valid_size = get_data_df_stare(shuffle=True)
    elif dataset_name == 'AV-WIDE':
        data_df, all_size, train_size, valid_size = get_data_df_av_wide(shuffle=True)
    elif dataset_name == 'CHASEDB':
        data_df, all_size, train_size, valid_size = get_data_df_chasedb(shuffle=True)
    elif dataset_name == 'VEVIO-FRAME':
        data_df, all_size, train_size, valid_size = get_data_df_vevio_frame(shuffle=True)
    elif dataset_name == 'VEVIO-MOSAICS':
        data_df, all_size, train_size, valid_size = get_data_df_vevio_mosaics(shuffle=True)
    else:
        return None
    return data_df, all_size, train_size, valid_size