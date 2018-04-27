#%matplotlib inline
import matplotlib
#%config InlineBackend.figure_format = 'retina'

import numpy as np
import time

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable

import helper

from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ])
# Download and load the training data
trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.MNIST('MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Defining the layers, 200, 50, 10 units each
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 50)
        # Output layer, 10 units - one for each digit
        self.fc3 = nn.Linear(50, 10)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        
        return x
    
    def predict(self, x):
        ''' This function for predicts classes by calculating the softmax '''
        logits = self.forward(x)
        return F.softmax(logits)

class ConvNet(nn.Module):
    def __init__(self, n_conv1=10, n_conv2=20):
        super().__init__()
        
        self.n_conv1, self.n_conv2 = n_conv1, n_conv2
        
        # conv layer with depth n_conv1, 5x5 kernels, and "same" padding
        self.conv1 = nn.Conv2d(1, n_conv1, 5, padding=2)
        # conv layer with depth n_conv2, 5x5 kernels, and "same" padding
        self.conv2 = nn.Conv2d(n_conv1, n_conv2, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        
        # The second convolutional layer will have size [7, 7, n_conv2], fc1 flattens it
        self.fc1 = nn.Linear(n_conv2*7*7, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        # First conv layer, then ReLU, then max-pooling
        x = F.relu(self.pool(self.conv1(x)))
        # Second conv layer, then ReLU, then max-pooling
        x = F.relu(self.pool(self.conv2(x)))
        # Flatten conv layer by reshaping
        x = x.view(-1, self.n_conv2*7*7)
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        #x = F.relu(self.fc3(x))
        
        return x
    
    def predict(self, x):
        logits = self.forward(x)
        return F.softmax(logits)


net = ConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

net.cuda()

trainloader.batch_size = 128
epochs = 5
steps = 0
running_loss = 0
print_every = 50
start0 = time.time()
for e in range(epochs):
    start = time.time()
    for images, labels in iter(trainloader):
        
        steps += 1

        inputs = Variable(images)
        targets = Variable(labels)
        
        inputs, targets = inputs.cuda(), targets.cuda()
        
        optimizer.zero_grad()
        
        output = net.forward(inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.data[0]
        
        if steps % print_every == 0:
            stop = time.time()
            # Test accuracy
            accuracy = 0
            for ii, (images, labels) in enumerate(testloader):
                
                inputs = Variable(images, volatile=True)
                inputs = inputs.cuda()
                
                # Get the class prediction and bring it back to the CPU
                predicted = net.predict(inputs).data.cpu()
                equality = (labels == predicted.max(1)[1])
                accuracy += equality.type_as(torch.FloatTensor()).mean()
            
            print("Epoch: {}/{}..".format(e+1, epochs),
                  "Loss: {:.4f}..".format(running_loss/print_every),
                  "Test accuracy: {:.4f}..".format(accuracy/(ii+1)),
                  "{:.4f} s/batch".format((stop - start)/print_every)
                 )
            running_loss = 0
            start = time.time()

## Save the model
filename = 'mnist1-1.ckpt'
checkpoint = {'n_conv1': net.n_conv1,
              'n_conv2': net.n_conv2,
              'state_dict': net.state_dict()}
with open(filename, 'wb') as f:
    torch.save(checkpoint, f)

## Load the model
filename = 'mnist1-1.ckpt'
with open(filename, 'rb') as f:
        checkpoint = torch.load(f)

model = ConvNet(n_conv1=checkpoint['n_conv1'],
                n_conv2=checkpoint['n_conv2'])
model.load_state_dict(checkpoint['state_dict'])

dataiter = iter(testloader)

print("total time: {:5.0f} s".format(time.time() - start0))

images, labels = dataiter.next()
img = images[0]
ps = model.predict(Variable(img.resize_(1, *img.size())))
helper.view_classify(img, ps)

