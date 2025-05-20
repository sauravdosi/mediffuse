import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import random
import torch


class CTMRDataset(Dataset):
    def __init__(self, data_dir, image_size=256, max_samples=2000):
        self.data_dir = data_dir
        self.image_size = image_size

        # Get all available ctcontour file names
        all_ids = [f.split('_')[0] for f in os.listdir(data_dir) if f.endswith('_ct.png')]

        # Randomly select max_samples
        random.seed(42)
        self.data_ids = random.sample(all_ids, min(max_samples, len(all_ids)))

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Now in [-1, 1] range
        ])

    # def __getitem__(self, idx):
    #     img_id = self.data_ids[idx]
    #     ct_img = Image.open(os.path.join(self.data_dir, f"{img_id}_ct.png")).convert("RGB")
    #     mr_img = Image.open(os.path.join(self.data_dir, f"{img_id}_mr.png")).convert("RGB")
    #
    #     return {
    #         "conditioning_image": self.transform(ct_img),  # Already a tensor
    #         "target_image": self.transform(mr_img)  # Already a tensor
    #     }

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        img_id = self.data_ids[idx]

        # Load CT + contour (input) and MR + contour (target)
        ct_path = os.path.join(self.data_dir, f"{img_id}_ct.png")
        mr_path = os.path.join(self.data_dir, f"{img_id}_mr.png")

        try:
            ct_img = Image.open(ct_path).convert("RGB")
            mr_img = Image.open(mr_path).convert("RGB")

            ct_tensor = self.transform(ct_img)  # [3, H, W]
            mr_tensor = self.transform(mr_img)

            return {
                "conditioning_image": ct_tensor,  # Input for ControlNet
                "target_image": mr_tensor  # Target MRI output
            }
        except Exception as e:
            print(f"Error loading pair {img_id}: {str(e)}")
            # Return a random tensor if file is corrupted/missing
            dummy = torch.rand(3, self.image_size, self.image_size)
            return {
                "conditioning_image": dummy,
                "target_image": dummy
            }

# Usage example:
# dataset = CTMRDataset("./data/ProcessedSlices", image_size=256, max_samples=2000)
# loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
#
# # Verification
# print(f"Total samples: {len(dataset)}")
# sample = dataset[0]
# print(f"CT shape: {sample['conditioning_image'].shape}")
# print(f"MR shape: {sample['target_image'].shape}")