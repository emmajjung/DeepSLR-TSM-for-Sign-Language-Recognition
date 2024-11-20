import json
import os
import cv2
import torch
from torch.utils.data import Dataset

class ASLDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        self.images = data['images']
        self.annotations = data['annotations']
        self.categories = {cat['id']: cat['name'] for cat in data['categories']}
        
        # Create a mapping from image_id to annotations
        self.image_to_annotations = {}
        for ann in self.annotations:
            image_id = ann['image_id']
            if image_id not in self.image_to_annotations:
                self.image_to_annotations[image_id] = []
            self.image_to_annotations[image_id].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to load image: {img_path}")
            image = torch.zeros((3, 224, 224), dtype=torch.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get annotations for this image
        anns = self.image_to_annotations.get(img_info['id'], [])
        
        # Process annotations
        if anns:
            bbox = anns[0]['bbox']  # Take the first bounding box
            category_id = anns[0]['category_id']
            label = self.categories[category_id]
        else:
            bbox = [0, 0, 0, 0]
            label = 'unknown'

        if self.transform:
            image = self.transform(image)

        # Convert label to index
        label_index = list(self.categories.values()).index(label)

        return image, label_index, torch.tensor(bbox, dtype=torch.float32)

# Usage:
# dataset = ASLDataset(root_dir='path/to/images', annotation_file='path/to/annotations.coco.json', transform=your_transform)