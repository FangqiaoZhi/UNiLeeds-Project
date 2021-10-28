import random
from torch.utils.data import Dataset
from PIL import Image

import time
import torch
import torchvision.transforms as tmf

tt = tmf.Compose([
    tmf.ToTensor()
])

from code.config import OUTPUT_SIZE, PADDING_SIZE, INPUT_SIZE
#OUTPUT_SIZE = 388
#PADDING_SIZE = 92

# 572 = 388 + 92 * 2
#INPUT_SIZE = OUTPUT_SIZE + PADDING_SIZE*2

top = torch.zeros(PADDING_SIZE, INPUT_SIZE)
left = torch.zeros(OUTPUT_SIZE, PADDING_SIZE)


def padding(p):
    return torch.cat((top, torch.cat((left, p, left),1), top), 0)

#from queue import Queue
def get_floodfill_result(label):
    # Step 1: Init global data
    n = OUTPUT_SIZE
    result = [[0 for p in range(n)] for q in range(n)]
    dx = [1, -1, 0, 0]
    dy = [0, 0, 1, -1]
    q = [[0 for p in range(n*n)], [0 for p in range(n*n)]]
    head = [0, 0]
    tail = [-1, -1]
    # End Step 1
    
    def _check_in_range(pi, pj):
        return pi >= 0 and pi < n and pj >=0 and pj < n
    def _floodfill(color):
        # check queue is not empty
        while head[color] <= tail[color]:
            # queue pop
            i, j = q[color][head[color]]
            head[color] = head[color] + 1
            for k in range(4):
                pi, pj = i + dx[k], j + dy[k]
                if _check_in_range(pi, pj):
                    if label[pi][pj] == color and result[pi][pj] == 0:
                        # queue push
                        tail[color] = tail[color] + 1
                        q[color][tail[color]] = (pi, pj)
                        result[pi][pj] = result[i][j] + 1 if label[i][j] > 0 else result[i][j] - 1
        
    def _get_floodfill():
        # Step 3: travel all pixels, to find all edge pixels
        for i in range(n):
            for j in range(n):
                flag = False
                # check there is different color in 4 neighbours
                for k in range(4):
                    pi, pj = i + dx[k], j + dy[k]
                    if _check_in_range(pi, pj):
                        if label[i][j] != label[pi][pj]:
                            flag = True
                            break
                if flag:
                    result[i][j] = 1 if label[i][j] > 0 else -1
                    color = label[i][j]
                    tail[color] = tail[color] + 1
                    q[color][tail[color]] = (i, j)
        # Step 4: floodfill not vessel
        _floodfill(0)
        # Step 5: floodfill vessel
        _floodfill(1)

    # Step 2: function entrance
    _get_floodfill()
    
    return torch.tensor(result)

class ImageLoader(Dataset):

    def __init__(self, df, floodfill=False, is_train=False, transform=None, label_transform=None):
        self.df = df
        #self.color_jitter = tmf.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
        #self.color_jitter = tmf.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
        self.color_jitter = tmf.ColorJitter()
        self.transform = [
            tmf.Compose([
                self.color_jitter,
                tmf.ToTensor()]),
            tmf.Compose([
                self.color_jitter,
                tmf.RandomHorizontalFlip(p=1), tmf.ToTensor()]),
            tmf.Compose([
                self.color_jitter,
                tmf.RandomVerticalFlip(p=1), tmf.ToTensor()]),
            tmf.Compose([
                self.color_jitter,
                tmf.RandomHorizontalFlip(p=1), tmf.RandomVerticalFlip(p=1), tmf.ToTensor()])
        ]
        self.label_transform = [
            tmf.Compose([tmf.ToTensor()]),
            tmf.Compose([tmf.RandomHorizontalFlip(p=1), tmf.ToTensor()]),
            tmf.Compose([tmf.RandomVerticalFlip(p=1), tmf.ToTensor()]),
            tmf.Compose([tmf.RandomHorizontalFlip(p=1), tmf.RandomVerticalFlip(p=1), tmf.ToTensor()])
        ]
        #self.label_transform = label_transform
        self.floodfill = floodfill
        self.is_train = is_train
        if not self.is_train:
            self.data_patches = []
            self.label_patches = []
            self.distanc_patches = []
            for index in range(len(self.df)):
                x, y = self.df['path'][index], self.df['class'][index]
                x = self.label_transform[0](x)
                y = self.label_transform[0](y)
                y = y[0]
                patches_x = []
                patches_y = []
                height, width = x.shape[1], x.shape[2]

                for p in range(0, height, OUTPUT_SIZE):
                    begin_p = p if p + OUTPUT_SIZE <= height else height - OUTPUT_SIZE
                    for q in range(0, width, OUTPUT_SIZE):
                        begin_q = q if q + OUTPUT_SIZE <= width else width - OUTPUT_SIZE
                        patch_x = x[0:3,begin_p:begin_p+OUTPUT_SIZE,begin_q:begin_q+OUTPUT_SIZE]
                        patch_x = torch.stack((
                            padding(patch_x[0]),
                            padding(patch_x[1]),
                            padding(patch_x[2])
                        ))
                        patch_y = y[begin_p:begin_p+OUTPUT_SIZE,begin_q:begin_q+OUTPUT_SIZE]
                        #print('pq', p, q, begin_p, begin_q, height, width)
                        self.data_patches.append(patch_x)
                        self.label_patches.append(patch_y)
                        if self.floodfill:
                            self.distanc_patches.append(get_floodfill_result(patch_y.int().tolist()))
                        else:
                            self.distanc_patches.append(0)
                
                #z = get_floodfill_result(y.int().tolist())
                #distanc_patches = [get_floodfill_result(y.int().tolist()) for y in patches_y]
                #return torch.stack(tuple(patches_x)), torch.stack(tuple(patches_y)),torch.stack(tuple(zs))

    def length(self):
        return len(self.df)

    def __len__(self):
        if self.is_train:
            return len(self.df)
        else:
            return len(self.data_patches)
        
    def __getitem__(self, index):
        #print('time', index, time.asctime(time.localtime(time.time())))
        #print('path', self.df['path'])
        #print('path', self.df['path'][index])
        '''
        x = Image.open(self.df['path'][index])
        try:
            x = x.convert('RGB') # To deal with some grayscale images in the data
        except:
            pass
        y = Image.open(self.df['class'][index])
        try:
            y = y.convert('RGB') # To deal with some grayscale images in the data
        except:
            pass
        '''
        
        if self.is_train:      
            x, y = self.df['path'][index], self.df['class'][index]
            transform_index = random.randint(0,3) if self.is_train else 0
            x = self.transform[transform_index](x)
            y = self.label_transform[transform_index](y)
            #print('x,y', x.shape, y.shape)
            
            height, width = x.shape[1], x.shape[2]
            patch_height = random.randint(0, height-OUTPUT_SIZE)
            patch_width = random.randint(0, width-OUTPUT_SIZE)        

            x = x[0:3,patch_height:OUTPUT_SIZE+patch_height,patch_width:OUTPUT_SIZE+patch_width]
            x = torch.stack((
                padding(x[0]),
                padding(x[1]),
                padding(x[2])
            ))
            y = y[0][patch_height:OUTPUT_SIZE+patch_height,patch_width:OUTPUT_SIZE+patch_width]
            

            if not self.floodfill:
                return x, y, 0

            z = get_floodfill_result(y.int().tolist())
            return x, y, z
        else:       
            return self.data_patches[index], self.label_patches[index], self.distanc_patches[index]
            '''
            x = self.label_transform[0](x)
            y = self.label_transform[0](y)
            y = y[0]
            patches_x = []
            patches_y = []
            
            
            height, width = x.shape[1], x.shape[2]
            for p in range(0, height, OUTPUT_SIZE):
                begin_p = p if p + OUTPUT_SIZE <= height else height - OUTPUT_SIZE
                for q in range(0, width, OUTPUT_SIZE):
                    begin_q = q if q + OUTPUT_SIZE <= width else width - OUTPUT_SIZE
                    patch_x = x[0:3,begin_p:begin_p+OUTPUT_SIZE,begin_q:begin_q+OUTPUT_SIZE]
                    patch_x = torch.stack((
                        padding(patch_x[0]),
                        padding(patch_x[1]),
                        padding(patch_x[2])
                    ))
                    patch_y = y[begin_p:begin_p+OUTPUT_SIZE,begin_q:begin_q+OUTPUT_SIZE]
                    #print('pq', p, q, begin_p, begin_q, height, width)
                    patches_x.append(patch_x)
                    patches_y.append(patch_y)
            if not self.floodfill:
                empty_list = [0 for p in range(len(patches_x))]
                return torch.stack(tuple(patches_x)), torch.stack(tuple(patches_y)), torch.Tensor(empty_list), len(patches_x)
            #z = get_floodfill_result(y.int().tolist())
            zs = [get_floodfill_result(y.int().tolist()) for y in patches_y]
            return torch.stack(tuple(patches_x)), torch.stack(tuple(patches_y)),torch.stack(tuple(zs)), len(patches_x)
            '''
    
    def get_patches(self, index):
        x, y = self.df['path'][index], self.df['class'][index]
        transform_index = 0
        x = self.transform[transform_index](x)
        y = self.label_transform[transform_index](y)
        y = y[0]
        patches_x = []
        patches_y = []
        
        height, width = x.shape[1], x.shape[2]
        for p in range(0, height, OUTPUT_SIZE):
            begin_p = p if p + OUTPUT_SIZE <= height else height - OUTPUT_SIZE
            for q in range(0, width, OUTPUT_SIZE):
                begin_q = q if q + OUTPUT_SIZE <= width else width - OUTPUT_SIZE
                patch_x = x[0:3,begin_p:begin_p+OUTPUT_SIZE,begin_q:begin_q+OUTPUT_SIZE]
                patch_x = torch.stack((
                    padding(patch_x[0]),
                    padding(patch_x[1]),
                    padding(patch_x[2])
                ))
                patch_y = y[begin_p:begin_p+OUTPUT_SIZE,begin_q:begin_q+OUTPUT_SIZE]

                patches_x.append(patch_x)
                patches_y.append(patch_y)
        return patches_x, patches_y, height, width, x, y
        
    def get_one_area(self, index):
        x, y = self.df['path'][index], self.df['class'][index]
        #transform_index = random.randint(0,3) if self.is_train else 0
        transform_index = 0
        x = self.transform[transform_index](x)
        y = self.label_transform[transform_index](y)
        y = y[0]
        
        height, width = x.shape[1], x.shape[2]
        patch_height = random.randint(0, height-OUTPUT_SIZE)
        patch_width = random.randint(0, width-OUTPUT_SIZE)        

        x = x[0:3,patch_height:OUTPUT_SIZE+patch_height,patch_width:OUTPUT_SIZE+patch_width]
        x = torch.stack((
            padding(x[0]),
            padding(x[1]),
            padding(x[2])
        ))
        y = y[patch_height:OUTPUT_SIZE+patch_height,patch_width:OUTPUT_SIZE+patch_width]
        

        if not self.floodfill:
            return x, y, 0

        z = get_floodfill_result(y.int().tolist())
        return x, y, z