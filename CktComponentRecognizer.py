# We shall define the neural network architecture and the prediction function in this separate file
# so that it can be used in other files seemlessly

# Libraries
from msilib.schema import Component
import torch.nn as nn
import torch
import cv2
from PIL import Image
from torchvision import transforms

tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


class myNet(nn.Module):
    def __init__(self):
        super().__init__()
        # using the AlexNet architecture
        # input images are already of size 224 x 224 x 3, no resizing required
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11,
                      stride=4, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(in_channels=96, out_channels=192,
                      kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(in_channels=192, out_channels=384,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=256,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256*6*6, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=10),
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.view(x.size(0), 256*6*6)
        x = self.classifier(x)

        return x


def recognize(img):
    '''
    The function recognizes the electronic component in the given image
    Input can be either:    string (specifying the file path to the image)
                            or
                            numpy.ndarray (Grayscale/RGB image)

    Output: Name of component recognized (str), Probability of the corresponding class

    '''
    components = ['battery', 'capacitor', 'diode', 'ground',
                  'inductor', 'led', 'mosfet', 'resistor', 'switch', 'transistor']

    if type(img) == str:
        img = cv2.imread(img)       # reading image from file path

    img = thresholdImage(img)       # converting image to binary
    img = resizeImage(img)          # resizing image to 224 x224 x 3
    # converting array to PIL image so it can be input to the model
    img = Image.fromarray(img)

    # initializing GPU, if present
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # loading trained model
    model = torch.load("./CktComponentRecognizer.pt")
    # shifting model to device
    model.to(device)

    with torch.no_grad():
        # converting PIL Image to tensor
        tf_img = tf(img).float().unsqueeze_(0).to(device)
        model.eval()
        output = model(tf_img)
        output = torch.max(output, dim=1)

    return components[output.indices.item()], output.values.item()


def thresholdImage(img):
    _, w = img.shape
    blockSize = w//5
    if blockSize % 2 == 0:
        blockSize += 1
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blockSize, 16)


def resizeImage(img):
    WIDTH = HEIGHT = 64
    return cv2.resize(img, [WIDTH, HEIGHT], interpolation=cv2.INTER_AREA)