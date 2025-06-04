# We shall define the neural network architecture and the prediction function in this separate file
# so that it can be used in other files seemlessly

# Libraries
import torch.nn as nn
import torch.nn.functional as F
import torch
import cv2
from PIL import Image
from torchvision import transforms
import os


tf = transforms.Compose([transforms.Grayscale(
    num_output_channels=1), transforms.ToTensor()])


class myNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.conv3 = nn.Conv2d(16, 24, 5)
        self.conv4 = nn.Conv2d(24, 32, 3)

        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(32*5*5, 320)
        self.fc2 = nn.Linear(320, 80)
        self.fc3 = nn.Linear(80, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x


def recognize_component(img):
    '''
    The function recognizes the electronic component in the given image
    Input can be either:    string (specifying the file path to the image)
                            or
                            numpy.ndarray (Grayscale/RGB image)

    Output: Name of component recognized (str), Probability of the corresponding class

    '''
    components = ['battery', 'capacitor', 'diode', 'ground',
                  'inductor', 'led', 'mosfet', 'resistor', 'switch', 'transistor']

    output_prob = 0

    if type(img) == str:
        img = cv2.imread(img, 0)       # reading image from file path
        img = thresholdImage(img)       # converting image to binary

    img = resizeImage(img)       # resizing to 84x84x1

    # initializing GPU, if present
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # loading trained model
    model = torch.load(r"./CktComponentRecognizer.pt")
    # shifting model to device
    model.to(device)

    with torch.no_grad():
        img = Image.fromarray(img)  # numpy.ndarray to PIL image
        tf_img = tf(img).float().unsqueeze_(0).to(
            device)  # converting PIL Image to tensor
        model.eval()
        output = model(tf_img)
        output = torch.max(output, dim=1)
        output_prob = float(output[0][0])
        output_name = components[output[1]]

    return output_name, output_prob


def thresholdImage(img):
    w = img.shape[1]
    blockSize = w//5
    if blockSize % 2 == 0:
        blockSize += 1
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blockSize, 16)


def resizeImage(img): 
    WIDTH = HEIGHT = 84
    return cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
