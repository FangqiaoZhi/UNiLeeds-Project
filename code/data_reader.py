
import code.model as model

import itertools
import numpy as np
import time
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision.transforms as tmf
import pandas as pd
from code.image_loader import ImageLoader
from code.config import DATASET_NAME


# Get value which is static in one running 
from code.static import lp_method
from code.dataset_prepare import get_data_frame


def get_loaders_and_datasets():
    data_df, all_size, train_size, valid_size = get_data_frame(DATASET_NAME)

    dataset_train = ImageLoader(
        df=data_df[:train_size-valid_size],
        is_train=True,
        floodfill=lp_method,
    )
    
    dataset_validate = ImageLoader(
        df=data_df[train_size-valid_size:train_size].reset_index(drop=True),
        is_train=False,
        floodfill=lp_method,
    )
    
    dataset_test = ImageLoader(
        df=data_df[train_size:].reset_index(drop=True),
        is_train=False,
        floodfill=True,
    )

    # Data loaders for use during training
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        #batch_size=72, 
        batch_size=2, 
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )
    validate_loader = torch.utils.data.DataLoader(
        dataset_validate,
        #batch_size=72, 
        batch_size=2, 
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        #batch_size=72, 
        batch_size=2, 
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    return (train_loader, validate_loader, test_loader), (dataset_train, dataset_validate, dataset_test)