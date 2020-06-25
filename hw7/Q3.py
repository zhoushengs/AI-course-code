import torch
import torch.nn as nn
from skimage import transform
import torchvision.transforms as transforms
from torch.autograd import Variable
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
#from vis_utils import *
import random
import math
import torchvision.datasets as datasets
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
#from utils import plot_images
import time
start =time.clock()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
        self.batch1 = nn.BatchNorm2d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
        self.batch2 = nn.BatchNorm2d(16, eps=1e-50, momentum=0.1, affine=True, track_running_stats=True)# 6*6 from image dimension

        self.conv3 = nn.Conv2d(16, 120, kernel_size=(5, 5), stride=(1, 1))
        self.batch3 = nn.BatchNorm2d(120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.fc1 = nn.Tanh()
        self.fc2 = nn.Linear(1080, 120, bias=True)
        self.fc3 = nn.Linear(120, 84, bias=True)
        self.fc4 = nn.Linear(84, 10, bias=True)

    def forward(self, x):
        # CNN layer
        x = self.fc1(self.batch1(self.conv1(x)))
        x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = self.fc1(self.batch2(self.conv2(x)))
        x = F.avg_pool2d(x, kernel_size=2, stride=1, padding=0)
        x = self.fc1(self.batch3(self.conv3(x)))

        # linear function
        x = x.view(x.shape[0], -1)
        x = self.fc2(x)
        x = self.fc1(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x





num_epochs = 8
num_class = 10
batch_size = 50
learning_rate = 0.001




transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
ini_train_data = datasets.FashionMNIST(root='./data',
                                   train=True,
                                   download=True,
                                   transform=transform
                                   )
test_data = datasets.FashionMNIST(root='./data',
                                  train=False,
                                  download=True,
                                  transform=transform)
# separate data to training data and validation data
lengths = [int(len(ini_train_data)*0.8), int(len(ini_train_data)*0.2)]
train_data, val_data = torch.utils.data.random_split(ini_train_data, lengths)





train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=batch_size,
                                           shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data,
                                         batch_size=batch_size,
                                         shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                        batch_size=4,
                                        shuffle=False)
sample_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                batch_size=5,
                                                shuffle=True,)




#t_image, t_label = next(iter(sample_loader))

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

# Show 5 sample of training data
# Get a batch of training data number = 4
#inputs, classes = next(iter(sample_loader))
# Make a grid from batch
#out = torchvision.utils.make_grid(inputs)
#torchvision.utils.save_image(out, 'samle.jpg')
#imshow(out)


cnn = Net()
cnn = cnn.to(device)
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
criterion = nn.CrossEntropyLoss()

epoch = 20

train_losses, val_losses = [],[]
for e in range(epoch):
    train_loss = 0
    val_loss = 0
    accuracy = 0
    train_accuracy = 0
    for data in train_loader:
        optimizer.zero_grad()
        images, train_labels = data[0].to(device),data[1].to(device)
        op = cnn(images)
        loss = criterion(op, train_labels)
        train_loss += loss.item()
        train_pro = torch.exp(op)
        train_top_probs, train_top_classes = train_pro.topk(1, dim=1)
        train_equals = train_labels == train_top_classes.view(train_labels.shape)
        train_accuracy += train_equals.type(torch.FloatTensor).mean()
        loss.backward()
        optimizer.step()
  #      print(loss)
    else:
        with torch.no_grad():
            #cnn.eval()
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                log_ps = cnn(images)
                prob = torch.exp(log_ps)
                top_probs, top_classes = prob.topk(1, dim=1)
                equals = labels == top_classes.view(labels.shape)
                accuracy += equals.type(torch.FloatTensor).mean()
                val_loss += criterion(log_ps, labels)
        cnn.train()
    print("Epoch: {}/{}.. ".format(e+1, epoch),
              "Training Loss: {:.3f}.. ".format(train_loss/len(train_loader)),
              "Training Accuracy: {:.3f}".format(train_accuracy/len(train_loader)),
              "Validation Loss: {:.3f}.. ".format(val_loss/len(val_loader)),
              "Validation Accuracy: {:.3f}".format(accuracy/len(val_loader)))
    train_losses.append(train_loss/len(train_loader))
    val_losses.append(val_loss/len(val_loader))



correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        t_image, t_label = data
        t_image, t_label = t_image.to(device), t_label.to(device)
        outputs = cnn(t_image)
        _, predicted = torch.max(outputs.data, 1)
        total += t_label.size(0)
        correct += (predicted == t_label).sum().item()

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

end = time.clock()
print('Running time: %s Seconds'%(end-start))
plt.plot(train_losses,label = "Train losses")
plt.plot(val_losses, label = "Test losses")
plt.legend()
plt.show()