import json
import os
import shutil
import random
import yaml
import cv2
from pathlib import Path
import albumentations as A
from tqdm import tqdm

# ================= Dataset Creation =================
def create_datasets(json_path, source_img_dir, output_base_dir):
    """Create initial datasets for balloons, goals, and combined"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    Path(output_base_dir).mkdir(parents=True, exist_ok=True)
    
    # Updated category mappings including combined
    category_maps = {
        "balloons": {0: 0, 4: 1},
        "goals": {5:0, 6:1, 7:2, 1:3, 2:4, 3:5},
        "combined": {0:0, 4:1, 5:2, 6:3, 7:4, 1:5, 2:6, 3:7}
    }

    # Create all three datasets
    dataset_paths = {}
    for dataset_type in ["balloons", "goals", "combined"]:
        # Create directories
        dataset_path = Path(output_base_dir) / dataset_type
        (dataset_path / "images").mkdir(parents=True, exist_ok=True)
        (dataset_path / "labels").mkdir(parents=True, exist_ok=True)
        
        # Process annotations
        valid_images = set()
        for ann in data['annotations']:
            if ann['category_id'] not in category_maps[dataset_type]:
                continue
                
            image_info = next(img for img in data['images'] if img['id'] == ann['image_id'])
            
            # Convert bbox to YOLO format
            x, y, w, h = ann['bbox']
            image_width = image_info['width']
            image_height = image_info['height']
            
            x_center = (x + w/2) / image_width
            y_center = (y + h/2) / image_height
            w /= image_width
            h /= image_height
            
            # Write label with remapped class ID
            label_path = dataset_path / "labels" / f"{Path(image_info['file_name']).stem}.txt"
            with open(label_path, 'a') as f:
                class_id = category_maps[dataset_type][ann['category_id']]
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
            
            # Copy image
            src_img = Path(source_img_dir) / image_info['file_name']
            dst_img = dataset_path / image_info['file_name']
            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)
            
            valid_images.add(dst_img)
        
        # Create dataset.yaml with combined classes
        yaml_content = {
            'path': str(dataset_path.resolve()),
            'train': 'images/train',
            'val': 'images/val',
            'names': {
                0: 'green_balloon',
                1: 'purple_balloon',
                2: 'yellow_circle',
                3: 'yellow_square',
                4: 'yellow_triangle',
                5: 'orange_circle',
                6: 'orange_square',
                7: 'orange_triangle'
            } if dataset_type == "combined" else (
                {
                    0: 'green_balloon',
                    1: 'purple_balloon'
                } if dataset_type == "balloons" else {
                    0: 'yellow_circle', 1: 'yellow_square', 2: 'yellow_triangle',
                    3: 'orange_circle', 4: 'orange_square', 5: 'orange_triangle'
                }
            )
        }
        
        with open(dataset_path / "dataset.yaml", 'w') as f:
            yaml.dump(yaml_content, f, sort_keys=False)
        
        dataset_paths[dataset_type] = dataset_path
    
    return dataset_paths

# ================= Augmentation =================
def create_augmentation_pipeline():
    """Create augmentation pipeline preserving primary colors"""
    return A.Compose([
        # Geometric transformations
        A.OneOf([
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.4),
            A.Affine(scale=(0.9, 1.1), translate_percent=0.05, rotate=(-10, 10), p=0.3)
        ], p=0.5),
        
        # Lighting variations (preserve hue)
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.3), contrast_limit=0.1, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5)
        ], p=0.7),
        
        # Environmental effects
        A.OneOf([
            A.GaussNoise(var_limit=(0.01, 0.05), p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.4),
            A.MotionBlur(blur_limit=(5, 9), p=0.4)
        ], p=0.6),
        
        # Optical effects
        A.OneOf([
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), src_radius=100, p=0.3),
            A.RandomShadow(num_shadows_lower=1, num_shadows_upper=3, p=0.3),
            A.CoarseDropout(max_holes=8, max_height=0.1, max_width=0.1, p=0.3)
        ], p=0.5),
        
        # Quality variations
        A.OneOf([
            A.ImageCompression(quality_range=(60, 80), p=0.4),
            A.Downscale(scale_range=(0.8, 0.95), p=0.4)
        ], p=0.4)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def augment_dataset(dataset_path, augmentations_per_image=50):
    """Perform augmentations and train/val split"""
    transform = create_augmentation_pipeline()
    img_dir = dataset_path / "images"
    label_dir = dataset_path / "labels"
    
    # Create augmented directories
    aug_img_dir = dataset_path / "aug_images"
    aug_label_dir = dataset_path / "aug_labels"
    aug_img_dir.mkdir(exist_ok=True)
    aug_label_dir.mkdir(exist_ok=True)
    
    # Process each image
    for img_path in tqdm(list(img_dir.glob("*.*")), desc=f"Augmenting {dataset_path.name}"):
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read original labels
        label_path = label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
            
        with open(label_path, 'r') as f:
            labels = []
            bboxes = []
            for line in f:
                class_id, x, y, w, h = map(float, line.strip().split())
                labels.append(int(class_id))
                bboxes.append([x, y, w, h])
        
        # Generate augmentations
        for i in range(augmentations_per_image):
            try:
                augmented = transform(image=image, bboxes=bboxes, class_labels=labels)
                aug_img_name = f"{img_path.stem}_aug{i}{img_path.suffix}"
                aug_label_name = f"{img_path.stem}_aug{i}.txt"
                
                # Save augmented image
                cv2.imwrite(
                    str(aug_img_dir / aug_img_name),
                    cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
                )
                
                # Save augmented labels
                with open(aug_label_dir / aug_label_name, 'w') as f:
                    for bbox, label in zip(augmented['bboxes'], augmented['class_labels']):
                        f.write(f"{label} {' '.join(f'{v:.6f}' for v in bbox)}\n")
                        
            except Exception as e:
                print(f"Error augmenting {img_path.name}: {str(e)}")
    
    # Merge original and augmented data
    all_images = list(img_dir.glob("*.*")) + list(aug_img_dir.glob("*.*"))
    all_labels = list(label_dir.glob("*.txt")) + list(aug_label_dir.glob("*.txt"))
    
    # Split into train/val (85-15)
    random.shuffle(all_images)
    split_idx = int(0.85 * len(all_images))
    
    # Create final directories
    final_img_dir = dataset_path / "images"
    final_label_dir = dataset_path / "labels"
    (final_img_dir / "train").mkdir(exist_ok=True)
    (final_img_dir / "val").mkdir(exist_ok=True)
    (final_label_dir / "train").mkdir(exist_ok=True)
    (final_label_dir / "val").mkdir(exist_ok=True)
    
    # Move files to final directories
    for img_path in all_images[:split_idx]:
        shutil.move(str(img_path), final_img_dir / "train" / img_path.name)
        label_path = (aug_label_dir if "aug" in img_path.stem else label_dir) / f"{img_path.stem}.txt"
        if label_path.exists():
            shutil.move(str(label_path), final_label_dir / "train" / label_path.name)
            
    for img_path in all_images[split_idx:]:
        shutil.move(str(img_path), final_img_dir / "val" / img_path.name)
        label_path = (aug_label_dir if "aug" in img_path.stem else label_dir) / f"{img_path.stem}.txt"
        if label_path.exists():
            shutil.move(str(label_path), final_label_dir / "val" / label_path.name)
    
    # Cleanup temporary directories
    shutil.rmtree(aug_img_dir)
    shutil.rmtree(aug_label_dir)

# ================= Main Execution =================
if __name__ == "__main__":
    # Step 1: Create initial datasets
    datasets = create_datasets(
        json_path='og_dataset/result.json',
        source_img_dir='og_dataset/',
        output_base_dir='datasets'
    )
    
    # Step 2: Augment and split each dataset
    for dataset_type, dataset_path in datasets.items():
        print(f"Processing {dataset_type} dataset...")
        augment_dataset(dataset_path, 50 if dataset_type == "balloons" else 75)
        #hardcoded for specific usecase but will add some kind of normalization across classes in the dataset
        
    print("Dataset creation and augmentation complete!")
