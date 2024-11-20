import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from ops.models import TSN

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
                non_local='simple',
                )
    
    # Load your pre-trained weights here
    # checkpoint = torch.load('path_to_your_model_weights.pth')
    # model.load_state_dict(checkpoint['state_dict'])
    
    model = model.to(device)
    model.eval()
    return model

# def preprocess_frame(frame):
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     frame = Image.fromarray(frame)
#     frame = transform(frame).unsqueeze(0)
#     return frame

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

def main():
    model = load_model()
    cap = cv2.VideoCapture(0)  # Use Mac camera

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess_frame(frame)
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            input_tensor = torch.autograd.Variable(input_tensor)
            output = model(input_tensor)

        _, predicted = torch.max(output, 1)
        gesture = gestures[predicted.item()]

        cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Hand Gesture Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
