import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from ops.models import TSN
from my_dataset import ASLDataset

# Check MPS availability
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    print("MPS device not found, using CPU instead.")

# Define the gestures (adjust according to your model's classes)
gestures = [
    "No gesture", "Swiping Left", "Swiping Right", "Swiping Down", "Swiping Up",
    "Pushing Hand Away", "Pulling Hand In", "Sliding Two Fingers Left",
    "Sliding Two Fingers Right", "Sliding Two Fingers Down", "Sliding Two Fingers Up",
    "Pushing Two Fingers Away", "Pulling Two Fingers In", "Rolling Hand Forward",
    "Rolling Hand Backward", "Turning Hand Clockwise", "Turning Hand Counterclockwise",
    "Zooming In With Full Hand", "Zooming Out With Full Hand", "Zooming In With Two Fingers",
    "Zooming Out With Two Fingers", "Thumb Up", "Thumb Down", "Shaking Hand",
    "Stop Sign", "Drumming Fingers", "No gesture", "Doing other things"
]

def load_model(num_class=27, num_segments=8):
    model = TSN(num_class, num_segments, 'RGB',
                base_model='resnet50',
                consensus_type='avg',
                img_feature_dim=256,
                pretrain='imagenet',
                is_shift=True, shift_div=8, shift_place='blockres',
                non_local='simple')
    
    model = model.to(device)
    model.eval()
    return model

def preprocess_frame(frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Resize to expected height and width
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Apply the transformation
    frame = transform(frame)
    
    # Reshape to [1, 8, 3, 224, 224]
    # 1 is the batch size, 8 is n_segment, 3 is the number of channels (RGB)
    frame = frame.unsqueeze(0).repeat(1, 8, 1, 1, 1)
    
    return frame

def train(model, train_loader, val_loader, num_epochs=10):
    criterion_cls = nn.CrossEntropyLoss()
    criterion_bbox = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        for images, labels, bboxes in train_loader:
            optimizer.zero_grad()
            
            images, labels, bboxes = images.to(device), labels.to(device), bboxes.to(device)
            
            # Original batch size from dataloader
            batch_size, channels, height, width = images.shape
            true_batch_size = batch_size // model.num_segments
            
            # Reshape images for the TSM model
            if batch_size % model.num_segments != 0:
                print(f"Skipping batch: {batch_size} not divisible by {model.num_segments}")
                continue
            
            # Forward pass
            try:
                class_output, bbox_output = model(images)
                print(f"Model outputs shapes:")
                print(f"class_output shape: {class_output.shape}")
                print(f"bbox_output shape: {bbox_output.shape}")
            except RuntimeError as e:
                print(f"Forward pass failed with shapes:")
                print(f"Images shape: {images.shape}")
                print(f"Labels shape: {labels.shape}")
                print(f"Bboxes shape: {bboxes.shape}")
                raise e

            # Compute loss
            loss_cls = criterion_cls(class_output, labels)
            loss_bbox = criterion_bbox(bbox_output, bboxes)
            loss = loss_cls + loss_bbox
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        # Validation logic similar to training
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for images, labels, bboxes in val_loader:
                images, labels, bboxes = images.to(device), labels.to(device), bboxes.to(device)
                
                batch_size = images.shape[0]
                true_batch_size = batch_size // model.num_segments
                
                if batch_size % model.num_segments != 0:
                    continue
                    
                images = images.view(true_batch_size, model.num_segments, channels, height, width)
                labels = labels.view(true_batch_size, -1)[:, 0]
                bboxes = bboxes.view(true_batch_size, -1, 4)[:, 0, :]
                
                class_output, bbox_output = model(images)
                val_loss += criterion_cls(class_output, labels).item() + criterion_bbox(bbox_output, bboxes).item()
                pred = class_output.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()

        val_loss /= len(val_loader)
        accuracy = correct / len(val_loader.dataset)
        print(f'Epoch {epoch+1}, Validation Loss: {val_loss}, Accuracy: {accuracy}')

    torch.save(model.state_dict(), 'asl_model.pth')

def freeze_bn(model):
    count = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            count += 1
            if count > 1:
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False

def main():
    # Define transforms
    print("start transform")
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print("end transformer")

    # Create datasets
    path_to_train_images = "train/images"
    path_to_train_annotations = "train/_annotations.coco.json"
    train_dataset = ASLDataset(path_to_train_images, path_to_train_annotations, transform=transform)
    print("end train dataset")

    path_to_val_images = "valid/images"
    path_to_val_annotations = "valid/_annotations.coco.json"
    val_dataset = ASLDataset(path_to_val_images, path_to_val_annotations, transform=transform)
    print("end val dataset")

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    print("end train_loader")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    print("end val_loader")

    # Load model
    model = load_model()
    print("end load model")

    freeze_bn(model)
    print("BatchNorm layers frozen except the first one")

    # Train model
    train(model, train_loader, val_loader)
    print("end train model")

    # After training, you can use the model for inference as before
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess_frame(frame)
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            class_output, bbox_output = model(input_tensor)
            _, predicted = torch.max(class_output, 1)
            gesture = gestures[predicted.item()]
            bbox = bbox_output.squeeze().cpu().numpy()

        # Draw bounding box and label on frame
        x, y, w, h = bbox
        cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
        cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Hand Gesture Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()