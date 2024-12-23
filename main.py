import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from ops.models import TSN
from dataset import ASLDataset
import os

asl_number_letter_map = {
    0: 'None', 1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J',
    11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R', 19: 'S', 20: 'T',
    21: 'U', 22: 'V', 23: 'W', 24: 'X', 25: 'Y', 26: 'Z', 
    }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_model(num_class=27):
    model = TSN(num_class, 'RGB',
                base_model='resnet50',
                pretrain='imagenet')
    model = model.to(device)
    model.eval()
    return model

def preprocess_frame(frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    frame = transform(frame)
    return frame.unsqueeze(0)

def train(model, train_loader, val_loader, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        print("epoch", epoch)
        model.train()
        for images, labels, _ in train_loader:
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            # print("outputs", outputs)
            loss = criterion(outputs, labels)
            print("loss", loss)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                pred = outputs.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()

        val_loss /= len(val_loader)
        accuracy = correct / len(val_loader.dataset)
        print(f'Epoch {epoch+1}, Validation Loss: {val_loss}, Accuracy: {accuracy}')

    torch.save(model.state_dict(), 'asl_model.pth')

def test_model_on_images(model, test_loader, show_images=False):
    correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for images, labels, file_paths in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Display images with labels and predictions
            if show_images:
                for i in range(images.size(0)):
                    img_path = file_paths[i]
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    #map letters and numbers: 
                    original_letter = asl_number_letter_map[labels[i].item()]
                    predicted_letter = asl_number_letter_map[predicted[i].item()]
                    
                    # show letter correspondance 
                    cv2.putText(img, f"Original: {original_letter}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 3)
                    cv2.putText(img, f"Predicted: {predicted_letter}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 3)
                    # create a window for the popup
                    cv2.namedWindow("Test Image", cv2.WINDOW_NORMAL)
                    ## change the window size to fit
                    cv2.resizeWindow("Test Image", 600, 800)
                    cv2.imshow("Test Image", img)
                    cv2.waitKey(0)

    if show_images:
        cv2.destroyAllWindows()
    accuracy = 100 * correct / total
    return accuracy

def main():
    # Define transforms
    print("Loading transformations...")
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create datasets
    print("Loading training dataset...")
    path_to_train_images = "train/images"
    path_to_train_annotations = "train/_annotations.coco.json"
    train_dataset = ASLDataset(path_to_train_images, path_to_train_annotations, transform=transform)

    print("Loading validation dataset...")
    path_to_val_images = "valid/images"
    path_to_val_annotations = "valid/_annotations.coco.json"
    val_dataset = ASLDataset(path_to_val_images, path_to_val_annotations, transform=transform)

    print("Loading test dataset...")
    path_to_test_images = "test/images"
    path_to_test_annotations = "test/_annotations.coco.json"
    test_dataset = ASLDataset(path_to_test_images, path_to_test_annotations, transform=transform)

    # Create data loaders
    print("Creating training dataset...")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    print("Creating validation dataset...")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    print("Creating test dataset...")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load model
    print("Loading model...")
    model = load_model()

    # Check if the trained model file exists
    model_path = "84_asl_model.pth"
    if os.path.exists(model_path):
        print("Loading pre-trained model...")
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
    else:
        print("Pre-trained model not found. Training new model...")
        train(model, train_loader, val_loader)

    # Test model: Change last parameter to True for showing images
    print("Testing model...")
    test_accuracy = test_model_on_images(model, test_loader, False)
    print(f"Test accuracy: {test_accuracy:.2f}%")

    open_front_camera = True
    model.eval()
    if open_front_camera:
        cap = cv2.VideoCapture(0)
        for i in range(2500):
            _, raw_image = cap.read()
            raw_image = cv2.flip(raw_image[:,600:1320,:], 1)
            if i%2 == 0:
                image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
                image = transform(image)
                image = image.to(device)
                image = image[None, :, :, :]
                output = model(image)
                _, predicted = torch.max(output.data, 1)
                cv2.putText(raw_image, f"Predicted: {asl_number_letter_map[predicted.item()]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
                cv2.imshow("Sign Language Recognition", raw_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()