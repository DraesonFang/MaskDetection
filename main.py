import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import xml.etree.ElementTree as ET


class MaskedFaceTestDataset(Dataset):
    def __init__(self, root, transform=None):
        super(MaskedFaceTestDataset, self).__init__()
        self.imgs = sorted(glob.glob(os.path.join(root, '*.png')))
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)


# Dataset for training with annotations
class MaskedFaceTrainingDataset(Dataset):
    def __init__(self, root, transform=None):
        super(MaskedFaceTrainingDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.imgs = sorted(glob.glob(os.path.join(root, '*.png')))

        # Class mapping
        self.class_map = {
            'with_mask': 0,
            'without_mask': 1,
            'mask_weared_incorrect': 2
        }

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = Image.open(img_path).convert("RGB")

        # Get corresponding XML annotation file
        xml_path = img_path.replace('.png', '.xml')

        # Parse XML to get bounding boxes and classes
        boxes = []
        labels = []

        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()

            for obj in root.findall('object'):
                name = obj.find('name').text
                if name in self.class_map:
                    class_id = self.class_map[name]

                    bbox = obj.find('bndbox')
                    xmin = float(bbox.find('xmin').text)
                    ymin = float(bbox.find('ymin').text)
                    xmax = float(bbox.find('xmax').text)
                    ymax = float(bbox.find('ymax').text)

                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(class_id)

        # Convert to tensor format required by Faster R-CNN
        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64)

        # Model expects background class to be 0, so add 1 to all class labels
        labels = labels + 1

        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([index]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.zeros(0),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }

        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

# Create a model for mask detection
def create_mask_detector_model(num_classes=4):  # 3 mask classes + background
    # Load pre-trained Faster R-CNN
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

    # Replace the classifier with a new one for our classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

# Train the model
def train_model(model, data_loader, optimizer, device, num_epochs=10, save_path='data/weights_counting.pth'):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Set model to training mode
    model.train()

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        running_loss = 0.0

        for images, targets in data_loader:
            # Move data to device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass and optimize
            losses.backward()
            optimizer.step()

            # Print statistics
            running_loss += losses.item()

        # Print epoch loss
        print(f"Epoch {epoch + 1} loss: {running_loss / len(data_loader):.4f}")

    # Save the model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    return model

# Evaluate the model
def evaluate_model(model, data_loader, device):
    model.eval()
    total_correct = 0
    total_objects = 0

    with torch.no_grad():
        for images, targets in data_loader:
            # Move data to device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Get predictions
            outputs = model(images)

            # Compare predictions with targets
            for i, (output, target) in enumerate(zip(outputs, targets)):
                # Apply confidence threshold
                keep = output['scores'] > 0.5
                pred_boxes = output['boxes'][keep]
                pred_labels = output['labels'][keep]

                # Get target boxes and labels
                gt_boxes = target['boxes']
                gt_labels = target['labels']

                # Simple evaluation - just count if we detected the same number of objects
                # A more sophisticated evaluation would use IoU matching
                total_objects += len(gt_labels)

                # Count correct class predictions (simplified)
                # This is a basic evaluation - real-world would use precision/recall/mAP
                if len(pred_labels) == len(gt_labels):
                    total_correct += 1

    accuracy = total_correct / len(data_loader) if len(data_loader) > 0 else 0
    print(f"Evaluation accuracy: {accuracy:.4f}")

    return accuracy

# Function to get ground truth counts from XML
def get_ground_truth_counts(xml_path):
    counts = np.zeros(3, dtype=np.int64)

    if os.path.exists(xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Class mapping (0-indexed)
        class_map = {
            'with_mask': 0,
            'without_mask': 1,
            'mask_weared_incorrect': 2
        }

        for obj in root.findall('object'):
            name = obj.find('name').text
            if name in class_map:
                class_idx = class_map[name]
                counts[class_idx] += 1

    return counts

# Calculate MAPE
def calculate_mape(true_counts, pred_counts):
    true_counts = np.array(true_counts)
    pred_counts = np.array(pred_counts)

    # Formula: MAPE = (1/n) * sum(|At - Pt| / max(At, 1)) * 100
    absolute_errors = np.abs(true_counts - pred_counts)
    denominators = np.maximum(true_counts, 1)  # Avoid division by zero
    percentage_errors = (absolute_errors / denominators) * 100

    # Average across all classes
    mape = np.mean(percentage_errors)

    return mape

# The main count_masks function
def count_masks(dataset):
    """
    Count the number of faces with masks, without masks, and incorrectly wearing masks.

    Args:
        dataset: An instance of MaskedFaceTestDataset

    Returns:
        counts: Numpy array of shape (3,) with counts for each class
        mape: Mean Absolute Percentage Error
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = create_mask_detector_model(num_classes=4)  # 3 mask classes + background

    # Load weights
    weights_path = 'data/weights_counting.pth'
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded model from {weights_path}")
    else:
        raise FileNotFoundError(f"Model weights not found at {weights_path}. Train the model first.")

    model.to(device)
    model.eval()

    # Initialize counters
    total_counts = []
    total_mape = 0.0

    # Process each image
    for i in range(len(dataset)):
        # Get image
        img = dataset[i]

        # Get image path and corresponding XML path
        img_path = dataset.imgs[i]
        xml_path = img_path.replace('.png', '.xml')

        # Get ground truth counts
        true_counts = get_ground_truth_counts(xml_path)

        # Prepare image for model
        if not isinstance(img, torch.Tensor):
            transform = transforms.Compose([transforms.ToTensor()])
            img = transform(img)

        # Add batch dimension
        input_tensor = img.unsqueeze(0).to(device)

        # Get predictions
        with torch.no_grad():
            predictions = model(input_tensor)

        # Process predictions
        boxes = predictions[0]['boxes'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()

        # Apply confidence threshold
        confidence_threshold = 0.5
        keep = scores > confidence_threshold
        labels = labels[keep]

        # Count detections by class
        image_counts = np.zeros(3, dtype=np.int64)
        for label in labels:
            # Convert from model labels (1-4) to our class indices (0-2)
            class_idx = label - 1
            if 0 <= class_idx < 3:
                image_counts[class_idx] += 1

        # Add to total counts
        total_counts.append(image_counts)

        # Calculate MAPE for this image
        image_mape = calculate_mape(true_counts, image_counts)
        total_mape += image_mape

    # Calculate average MAPE
    avg_mape = total_mape / len(dataset) if len(dataset) > 0 else 0.0

    return np.array(total_counts, dtype=np.int64), avg_mape

def collate_fn(batch):
    return tuple(zip(*batch))

# Main function to run the entire pipeline
def main():
    # Data paths
    train_data_root = "maskdata/train"  # Path to training data
    test_data_root = "maskdata/val"  # Path to test data

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Create datasets
    train_dataset = MaskedFaceTrainingDataset(root=train_data_root, transform=transform)
    test_dataset = MaskedFaceTestDataset(root=test_data_root, transform=transform)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn
    )

    # Create model
    model = create_mask_detector_model(num_classes=4)  # 3 mask classes + background
    model.to(device)

    # Define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Check if model is already trained
    weights_path = 'data/weights_counting.pth'
    if os.path.exists(weights_path):
        print(f"Loading pre-trained model from {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print("Training new model...")
        model = train_model(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            num_epochs=100,
            save_path=weights_path
        )

    # Count masks in test dataset
    print("Counting masks in test dataset...")
    counts, mape = count_masks(test_dataset)

    print(f"Final counts: {counts}")
    print(f"MAPE: {mape:.2f}%")


if __name__ == "__main__":
    main()
