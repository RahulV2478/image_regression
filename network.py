#source: https://github.com/hugohadfield/pytorch_image_regession
from typing import Tuple

import torch.nn as nn
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import models

from torch.utils.tensorboard import SummaryWriter

from data_loading import RegressionTaskData


class CNNRegression(nn.Module):
    """
    This will be the very basic CNN model we will use for the regression task.
    """
    def __init__(self, image_size: Tuple[int, int, int] = (3, 100, 100)):
        super(CNNRegression, self).__init__()
        self.image_size = image_size
        self.conv1 = nn.Conv2d(in_channels=self.image_size[0], out_channels=4, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear_line_size = int(16*(image_size[1]//4)*(image_size[2]//4))
        self.fc1 = nn.Linear(in_features=self.linear_line_size, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=4)

        
    def forward(self, x):
        """
        Passes the data through the network.
        There are commented out print statements that can be used to 
        check the size of the tensor at each layer. These are very useful when
        the image size changes and you want to check that the network layers are 
        still the correct shape.
        """
        x = self.conv1(x)
        # print('Size of tensor after each layer')
        # print(f'conv1 {x.size()}')
        x = nn.functional.relu(x)
        # print(f'relu1 {x.size()}')
        x = self.pool1(x)
        # print(f'pool1 {x.size()}')
        x = self.conv2(x)
        # print(f'conv2 {x.size()}')
        x = nn.functional.relu(x)
        # print(f'relu2 {x.size()}')
        x = self.pool2(x)
        # print(f'pool2 {x.size()}')
        x = x.view(-1, self.linear_line_size)
        # print(f'view1 {x.size()}')
        x = self.fc1(x)
        # print(f'fc1 {x.size()}')
        x = nn.functional.relu(x)
        # print(f'relu2 {x.size()}')
        x = self.fc2(x)
        # print(f'fc2 {x.size()}')
        #print(x)
        return x

class ResNetCNNRegression(nn.Module):
    def __init__(self):
        super(ResNetCNNRegression, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()
        self.regressor = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128,4)
        )
    def forward(self, x):
        features = self.backbone(x)
        output = self.regressor(features)
        return output
    
#Source: https://huggingface.co/blog/tonyassi/image-regression
#Source: https://huggingface.co/google/vit-base-patch16-224?library=transformers
#Source: https://huggingface.co/docs/transformers/main/en/model_doc/vit

#Source: https://github.com/TonyAssi/ImageRegression/blob/main/ImageRegression.py

#Source: https://github.com/huggingface/pytorch-image-models/discussions/2030

from transformers import ViTModel
import timm
class ViTRegressionModel(nn.Module):
    def __init__(self):
        super(ViTRegressionModel, self).__init__()
        #self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.vit = timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=True, img_size=(1500, 2000))
        # print(self.vit.config.hidden_size) 768
        self.classifier = nn.Linear(768, 1)

    def forward(self, x, labels=None):
        outputs = self.vit(x)
        cls_output = outputs.last_hidden_state[:, 0, :]  # Take the [CLS] token
        values = self.classifier(cls_output)
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(values.view(-1), labels.view(-1))
        return (loss, values) if loss is not None else values


def train_network(device, n_epochs: int = 10, image_size: Tuple[int, int, int] = (3, 100, 100)):
    """
    This trains the network for a set number of epochs.
    """
    if image_size[0] == 1:
        grayscale = True
    else:
        grayscale = False
    # assert image_size[1] == image_size[2], 'Image size must be square'
    # resize_size = image_size[1]
    # regression_task = RegressionTaskData(grayscale=grayscale, resize_size=resize_size)
    regression_task = RegressionTaskData(grayscale=grayscale)
    regression_task.visualise_image(5)

    # Define the model, loss function, and optimizer
    model = CNNRegression(image_size=image_size)
    # model = ResNetCNNRegression()
    #model = ViTRegressionModel()
    model.to(device)
    print(model)
    criterion = nn.MSELoss()
    # criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    writer = SummaryWriter()
    for epoch in range(n_epochs):
        for i, (inputs, targets) in enumerate(regression_task.trainloader):
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs.to(device))
            loss = criterion(outputs, targets.to(device))

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            writer.add_scalar('Train Loss', loss.item(), i)

            # Print training statistics
            #if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{n_epochs}], Step [{i + 1}/{len(regression_task.trainloader)}], Loss: {loss.item():.4f}')
    writer.close()

    return model


def save_model(model, filename='3_100_100.pth'):
    """
    After training the model, save it so we can use it later.
    """
    torch.save(model.state_dict(), filename)


def load_model(image_size=(3, 100, 100), filename='200epochs_1500_2000.pth'):
    """
    Load the model from the saved state dictionary.
    """
    model = CNNRegression(image_size=image_size)
    #model = ResNetCNNRegression()
    model.load_state_dict(torch.load(filename, map_location=torch.device("cpu")))
    return model


def evaluate_network(model, device, image_size: Tuple[int, int, int] = (3, 1500, 2000)):
    """
    This evaluates the network on the test data.
    """
    if image_size[0] == 1:
        grayscale = True
    else:
        grayscale = False
    # assert image_size[1] == image_size[2], 'Image size must be square'
    # resize_size = image_size[1]
    # regression_task = RegressionTaskData(grayscale=grayscale, resize_size=resize_size)
    regression_task = RegressionTaskData(grayscale=grayscale)
    criterion = nn.MSELoss()

    # Evaluate the model on the test data
    with torch.no_grad():
        total_loss = 0
        total_angle_error = 0
        n_samples_total = 0
        length = 0
        print(regression_task.testloader)
        for inputs, targets in regression_task.testloader:
            length = len(inputs)
            # Calculate the loss with the criterion we used in training
            outputs = model(inputs.to(device))
            print("targets below:")
            print(targets)

            # image = Image.open(io.BytesIO(source_bytes))
            # draw = ImageDraw.Draw(image)
            # draw.line(points, width=5, fill = "#69f5d9")

            # shape = [(left - 2, top - 35), (width + 2 + left, top)]
            # draw.rectangle(shape, fill = "#69f5d9")

            # font = ImageFont.load_default()

            print("here")
            print(outputs)
            loss = criterion(outputs, targets.to(device))
            total_loss += loss.item()

            # We are actually predicting angles so we can calculate the angle error too
            # which is probably more meaningful to humans that the MSE loss
            # outputs_np = outputs.cpu().numpy()
            # targets_np = targets.cpu().numpy()
            # output_angles = np.array([np.arctan2(out[0], out[1]) for out in outputs_np])
            # target_angles = np.array([np.arctan2(t[0], t[1]) for t in targets_np])
            # This is probably not a great way to calculate the angle error 
            # as it doesn't take into account the fact that angles wrap around
            # but it seems to work well enough for now
            # angle_error = np.sum(np.abs(np.rad2deg(target_angles - output_angles)))
            # total_angle_error += angle_error
            # n_samples_total += len(output_angles)

        mean_loss = total_loss / len(regression_task.testloader)
        #mean_angle_error = total_angle_error / n_samples_total
        print(f'Test Loss: {mean_loss:.4f}')
        #print(f'Test mean angle error: {mean_angle_error:.4f} degrees')

        
        for i in range(length):
            fig, ax = plt.subplots()
            response = outputs[i]
            true_response = targets[i]
            ax.imshow(inputs[i].permute(1, 2, 0))
            print(response[0], response[1], response[2], response[3])
            rect = patches.Rectangle((response[0], response[1]), response[2], response[3], linewidth=1, edgecolor='r', facecolor='none')
            true = patches.Rectangle((true_response[0], true_response[1]), true_response[2], true_response[3], linewidth=1, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
            ax.add_patch(true)
            plt.show()



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Train the model
    image_size: Tuple[int, int, int] = (3, 1500, 2000)
    #model = train_network(device, 50, image_size=image_size)

    # Save the model
    filename = '200epochs_1500_2000.pth'
    #save_model(model, filename=filename)

    # Load the model
    model = load_model(image_size=image_size, filename=filename)
    model.to(device)

    # Evaluate the model
    evaluate_network(model, device, image_size=image_size)