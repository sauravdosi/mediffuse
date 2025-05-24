import json
import cv2
import numpy as np

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('/home/sauravdosi/mediffuse/data/train_contour2img/metadata.jsonl', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['input_image']
        target_filename = item['edited_image']
        prompt = item['edit_prompt']

        source = cv2.imread('/home/sauravdosi/mediffuse/data/train_contour2img/' + source_filename)
        target = cv2.imread('/home/sauravdosi/mediffuse/data/train_contour2img/' + target_filename)

        # Resize both images to a fixed size (e.g., 256x256)
        target_size = (512, 512)
        source = cv2.resize(source, target_size, interpolation=cv2.INTER_AREA)
        target = cv2.resize(target, target_size, interpolation=cv2.INTER_AREA)

        # Convert BGR to RGB
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize
        source = source.astype(np.float32) / 255.0
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

