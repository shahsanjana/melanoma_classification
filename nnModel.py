# %%
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import sys 

# %%
# 50 X 50 pixels
img_size = 50

#how to define input and output size? 
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        #convolutional layers
        self.conv1 = nn.Conv2d(1, 32,5)
        self.conv2 = nn.Conv2d(32, 64,5)
        self.conv3 = nn.Conv2d(64, 128,5)

        #linear layers
        #FIXME - Get input size
        self.fc1 = nn.Linear(128*2*2, 512)
        self.fc2 = nn.Linear(512, 2)  # 2 output classes

    #x is the input image    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2)) #(2,2) is kernal size
        # print(f"cv 1 {x.shape}")
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2)) 
        # print(f"cv 2 {x.shape}")
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2)) 
        # print(f"cv 3 {x.shape}")
        # sys.exit("get shape for linear layer")
        x = x.view(-1, 128*2*2)  #flattening
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x)  #probablity output
        return x


#net = Net()

# %%
# test_img = torch.randn(img_size, img_size).view(-1,1, img_size, img_size)  #batch size, channels, height, width
# output = net(test_img)
# print(output)
# cv 1 torch.Size([1, 32, 23, 23])
# cv 2 torch.Size([1, 64, 9, 9])
# cv 3 torch.Size([1, 128, 2, 2])



