import os
import torch
from torch.utils.data import Dataset, DataLoader
# import glob
from torchvision import transforms
import numpy as np  
from PIL import Image

from common import representation
from common import event
from labits import calc_labits


# import random


class EventVoxel(Dataset):
    def __init__(self, data_root, is_hsergb, is_training, is_validation, number_of_time_bins, number_of_skips=1):
        """
            Aims to create an EventVoxel object.
            Inputs,
                data_root: Root path for the dataset
                is_hsergb: Indicator for the event-frame dataset False if bs-ergb dataset intedted to be used.
                is_training: Indicates whether the dataset is for training.
                is_validation: Indicates whether the dataset is for validation. ( works if is_training set to false)
                number_of_time_bins: Indicates TWO of the event files' temporal seperation in voxel 
                number_of_skips: No functionality for this version of the class.
            Outputs,
                images = 4 image with 2 leftmost 2 rightmost frame neighbors of the ground truth ( in 4x3x256x256 )
                voxel grid = 12 channel voxel grid representation of 2 leftmost 2 rightmost event neighbors of the ground truth ( in 4x3x256x256 )        
                ground truth = ground truth image ( in 3x256x256 )
        """
        self.data_root = data_root
        self.is_training = is_training  # if True it is training set
        self.is_validation = is_validation  # if True it is validation set
        self.is_hsergb = is_hsergb
        self.number_of_time_bins = number_of_time_bins
        self.number_of_skips = number_of_skips

        if is_hsergb:
            train_fn = os.path.join(self.data_root, 'data_set_hsergb.txt')
        else:
            train_fn = os.path.join(self.data_root, 'set_data_bsergb_training.txt')
            test_fn = os.path.join(self.data_root, 'set_data_bsergb_test.txt')
            valid_fn = os.path.join(self.data_root, 'set_data_bsergb_validation.txt')

        if is_hsergb:
            with open(train_fn, 'r') as f:
                self.trainlist = f.read().splitlines()  # TODO split to a training set
        else:
            with open(train_fn, 'r') as f:
                self.trainlist = f.read().splitlines()
            with open(test_fn, 'r') as f:
                self.testlist = f.read().splitlines()
            with open(valid_fn, 'r') as f:
                self.validationlist = f.read().splitlines()

        self.transforms = transforms.Compose([
           transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])


    def __getitem__(self, index):

        """
        Every pass returns an image of ground truth , 4 events (2 normal 2 reversed in a voxel of 12 bins) and right,left images
        Total 3 images with 9 channel 1 voxel of 12 channel is obtained.
        """

        if self.is_hsergb:
            raw_data = self.trainlist[index].split(" ")
        else:
            if self.is_training:
                raw_data = self.trainlist[index].split(" ")
            elif self.is_validation:
                raw_data = self.validationlist[index].split(" ")
            else:
                raw_data = self.testlist[index].split(" ")
        raw_data.pop()  # this is for the blank at the end of a line
        raw_data = [os.path.join(self.data_root, raw_) for raw_ in raw_data]
        list_of_images = raw_data[:4]
        gt_path = raw_data[4]
        list_of_events = raw_data[5:]

        # the_pair = random.randint(2, len(list_of_events)-3) #Since the 5 image 4 files of events are selected last 2 and first 2 can not be the ground truth image ( center image to be predicted )

        gt = Image.open(gt_path)  # since events are sparse an image is fetched in order to have HxW information.

        W, H = gt.size

        events_left = event.EventSequence.from_npz_files(
            list_of_filenames=list_of_events[:3],
            image_height=H,
            image_width=W,
            hsergb=self.is_hsergb,
        )

        # our modification  
        # voxel_left = representation.to_voxel_grid(events_left, nb_of_time_bins=self.number_of_time_bins)
        
        # Convert timestamps to [0, nb_of_time_bins] range.
        duration = events_left.duration()
        start_timestamp = events_left.start_time()
        features = torch.from_numpy(events_left._features)
        x = features[:, event.X_COLUMN]
        y = features[:, event.Y_COLUMN]
        polarity = features[:, event.POLARITY_COLUMN].float()
        t = (features[:, event.TIMESTAMP_COLUMN] - start_timestamp) * (self.number_of_time_bins - 1)/ duration * 10
        t = t.float()
        t_span = t[-1] - t[0]
        t_range = t_span / (self.number_of_time_bins+1)
        
        h=events_left._image_height,
        w=events_left._image_width,

        x = np.array(x) if not isinstance(x, np.ndarray) else x
        y = np.array(y) if not isinstance(y, np.ndarray) else y
        t = np.array(t) if not isinstance(t, np.ndarray) else t
        
        t_range= np.array(t_range) if not isinstance(t_range, np.ndarray) else t_range
        # round t_range and change it to int
        t_range=int(np.round(t_range))
        voxel_left = calc_labits(xs=x, ys=y, ts=t, framesize=(h[0],w[0]), t_range=t_range, num_bins=self.number_of_time_bins+1, norm=True)[1:]
        
        
        

        events_right = event.EventSequence.from_npz_files(
            list_of_filenames=list_of_events[3:],
            image_height=H,
            image_width=W,
            hsergb=self.is_hsergb,
        )
        events_right.reverse()


        # our modification
        # voxel_right = representation.to_voxel_grid(events_right, nb_of_time_bins=self.number_of_time_bins)
        
        duration = events_right.duration()
        start_timestamp = events_right.start_time()
        features = torch.from_numpy(events_right._features)
        x = features[:, event.X_COLUMN]
        y = features[:, event.Y_COLUMN]
        polarity = features[:, event.POLARITY_COLUMN].float()
        t = (features[:, event.TIMESTAMP_COLUMN] - start_timestamp) * (self.number_of_time_bins - 1) / duration*10
        t = t.float()
        t_span = t[-1] - t[0]
        t_range = t_span / (self.number_of_time_bins+1)
        
        h=events_right._image_height,
        w=events_right._image_width,

        x = np.array(x) if not isinstance(x, np.ndarray) else x
        y = np.array(y) if not isinstance(y, np.ndarray) else y
        t = np.array(t) if not isinstance(t, np.ndarray) else t
        t_range= np.array(t_range) if not isinstance(t_range, np.ndarray) else t_range
        # round t_range and change it to int
        t_range=int(np.round(t_range))
        voxel_right = calc_labits(xs=x, ys=y, ts=t, framesize=(h[0],w[0]), t_range=t_range, num_bins=self.number_of_time_bins+1, norm=True)[1:]
        

        voxel = torch.cat((voxel_left, voxel_right))

        images = [Image.open(pth) for pth in list_of_images]

        T = self.transforms

        images = [T(img_) for img_ in images]
        gt = T(gt)

        voxel = torch.reshape(voxel, (4, self.number_of_time_bins // 2, H, W))  # reshape to have a 4x3xHxW tensor
        #V = transforms.Compose([transforms.ToPILImage(), transforms.Resize((256, 256)),
        # transforms.ToTensor()])  # Voxel grid transforms for 256 #TODO random crop size is subject to change.

        voxel = torch.nn.functional.interpolate(voxel, size=(256, 256))
        # V = transforms.Compose([transforms.ToTensor()])  # Voxel grid transforms for 256 #TODO random crop size is subject to change.

        #voxel = [V(vox_) for vox_ in voxel[:]]
        voxel = list(voxel)
        return images, voxel, gt ,gt_path # note that all three are torch tensor

    def __len__(self):
        if self.is_training:
            return len(self.trainlist)
        elif self.is_validation:
            return len(self.validationlist)
        else:
            return len(self.testlist)


# def get_loader(mode, data_root, batch_size, shuffle, num_workers, test_mode=None):
#     if mode == 'train':
#         is_training = True
#     else:
#         is_training = False
#     dataset = EventVoxel(data_root, is_training=is_training)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


if __name__ == "__main__":
    dataset = EventVoxel("../BS-ERGB", is_hsergb=False, is_training=True, is_validation=False, number_of_time_bins=6)
    print(dataset[0])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
