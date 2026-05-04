"""
ImageNet100 dataset class for use with PyTorch DataLoader.

Loads images from disk, applies a standard ImageNet preprocessing pipeline,
and returns (preprocessed_tensor, class_index) tuples.
"""
from pathlib import Path
import json
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# Standard ImageNet preprocessing pipeline
# Used for evaluation (deterministic; no augmentation)
IMAGENET_PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class ImageNet100Dataset(Dataset):
    """
    Loads ImageNet-100 images and returns (preprocessed_tensor, class_index) tuples.
    
    Args:
        items: list of (image_path, synset_id) tuples
        synset_to_idx: dict mapping synset_id (str) to class index (int)
        transform: optional torchvision transform to apply
    """
    def __init__(self, items, synset_to_idx, transform=None):
        self.items = items
        self.synset_to_idx = synset_to_idx
        self.transform = transform
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        img_path, synset = self.items[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = self.synset_to_idx[synset]
        return image, label


def collect_files(split_folders):
    """Walk one or more folders and collect (path, synset_id) tuples."""
    items = []
    for folder in split_folders:
        if not folder.exists():
            print(f"Warning: {folder} does not exist, skipping")
            continue
        for class_dir in sorted(folder.iterdir()):
            if not class_dir.is_dir():
                continue
            synset = class_dir.name
            for img_path in class_dir.iterdir():
                if img_path.suffix.upper() == ".JPEG":
                    items.append((img_path, synset))
    return items


def load_labels(data_root):
    """Load Labels.json from the dataset root."""
    with open(Path(data_root) / "Labels.json") as f:
        return json.load(f)