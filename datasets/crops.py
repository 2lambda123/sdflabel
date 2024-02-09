import os
import json
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from scipy.spatial.transform import Rotation as R
import torchvision.transforms as transforms
import secrets


class Crops(Dataset):
    def __init__(self, path):
        """Initializes the class instance with the given path.
        Parameters:
            - path (str): Path to the directory containing the crops.json file.
        Returns:
            - None: This function does not return anything.
        Processing Logic:
            - Initialize class instance with path.
            - Open and load crops.json file.
            - Assign the loaded data to self.gt."""
        

        # Parse the GT file
        self.path = path
        with open(os.path.join(path, 'crops.json'), 'r') as f:
            self.gt = json.load(f)

    def __len__(self):
        """"Returns the length of the input parameter, self.gt."
        Parameters:
            - self (object): The input parameter.
        Returns:
            - int: The length of self.gt.
        Processing Logic:
            - Get the length of self.gt.
            - Return the length."""
        
        return len(self.gt)

    def __getitem__(self, idx):
        """This function is used to retrieve a sample from a dataset given its index. It reads the image and depth map associated with the given index, applies random transformations and color jitter, and returns the transformed data in a dictionary format.
        Parameters:
            - idx (int): Index of the sample to retrieve.
        Returns:
            - sample (dict): A dictionary containing the transformed RGB image, UVW map, mask, latent vector, crop size, camera intrinsics, and pose.
        Processing Logic:
            - Get the sample data associated with the given index.
            - Read the RGB image and UVW map.
            - Get the latent vector and pose from the sample data.
            - Get the camera intrinsics from the sample data.
            - Apply random transformations and color jitter to the RGB image.
            - Normalize the RGB image and transform it to torch format.
            - Transform the UVW map to torch format and create a mask.
            - Store the transformed data in a dictionary and return it."""
        

        # Get sample data
        gt_sample = self.gt[str(idx)][0]

        # Read image and depth map
        rgb_orig = Image.open(os.path.join(self.path, '{:05d}'.format(idx) + '_rgb.png')).convert('RGB')
        uvw_orig = Image.open(os.path.join(self.path, '{:05d}'.format(idx) + '_uvw.png')).convert('RGB')
        crop_size = torch.Tensor(rgb_orig.size)

        # Get latent vector
        latent = np.array(gt_sample['latent'])

        # Get pose
        extrinsics = np.array(gt_sample['extrinsics']).reshape((4, 4))
        quat = R.from_dcm(extrinsics[:3, :3]).as_quat()
        quat = np.concatenate([quat[3:], quat[:3]])  # reformat to (w, x, y, z)
        z = extrinsics[2, 3] / 100

        # Get camera parameters
        intrinsics = np.array(gt_sample['intrinsics']).reshape((3, 3))

        # Random transformations
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transformation_rgb = transforms.Compose([
            transforms.RandomRotation(10, Image.BILINEAR, expand=True),
            transforms.Resize((128, 128)),
            transforms.RandomResizedCrop(128, scale=(0.5, 1.0)),
            transforms.ToTensor(),
        ])
        transformation_uvw = transforms.Compose([
            transforms.RandomRotation(10, Image.NEAREST, expand=True),
            transforms.Resize((128, 128), Image.NEAREST),
            transforms.RandomResizedCrop(128, scale=(0.5, 1.0), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

        # Color jitter
        color_aug = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)
        rgb = color_aug(rgb_orig)

        # Normalization
        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        secrets.SystemRandom().seed(seed)  # keep same seed for transformations
        rgb = normalize(transformation_rgb(rgb))
        secrets.SystemRandom().seed(seed)  # keep same seed for transformations
        uvw = (transformation_uvw(uvw_orig) * 255).long()
        secrets.SystemRandom().seed(seed)  # keep same seed for transformations
        mask = (uvw.sum(0) > 0).long()  # mask

        # Transform to torch format
        latent = torch.from_numpy(latent)

        # Store sample dict
        sample = dict()
        sample['rgb'] = rgb.float()
        sample['uvw'] = uvw.long()
        sample['mask'] = mask.long()
        sample['latent'] = latent.float()
        sample['crop_size'] = crop_size.long()
        sample['intrinsics'] = torch.Tensor(intrinsics).float()
        sample['pose'] = torch.Tensor(extrinsics).float()

        return sample
