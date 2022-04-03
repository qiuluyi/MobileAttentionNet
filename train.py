import datetime as dt
import os
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import *
from model import MyModel
from mobilenet import *

data_path = "./dataset/"
img_path = os.path.join(data_path, 'images')
lab_path = os.path.join(data_path, 'labels')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 1
epochs = 50
# ToTensor：convert a PIL image to tensor (H*W*C) in range [0,255] to a torch.Tensor(C*H*W) in the range [0.0,1.0]
transform = transforms.Compose([transforms.ToTensor(), ])
net = MobileNet().to(device)
train_dataset = MyDataset(images_path=img_path, labels_path=lab_path, transform=transform)
dataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
lossFuction = nn.BCELoss()
lr = 0.001
momentum = 0.8
optimizer = optim.Adam(net.parameters(), lr=lr)


def shift_time_to_path(datetime):
    path = '_'.join(datetime.split())
    path = '_'.join(path.split(':'))
    path = os.path.join('./checkpoints', path)
    return path


def train():
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')

    now_time = dt.datetime.now().strftime('%F %T')
    save_path = shift_time_to_path(now_time)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    net.train()
    for epoch in range(1, epochs + 1):
        start = time.clock()
        for batch_indx, (img, label) in enumerate(dataLoader):

            img, label = img.to(device), label.to(device)
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            output = torch.sigmoid(net(img))
            loss = lossFuction(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_indx % 4 == 0:
                print('Epoch:[{}/{}]\tStep:[{}/{}]\tLoss:{:.6f}\tTime:{}'.format(
                    epoch, epochs, batch_indx * batch_size + len(img), len(dataLoader.dataset), loss.item(),
                                   time.clock() - start
                ))
                start = time.clock()
        if epoch % 10 == 0:
            save_name = os.path.join(save_path, 'mobilenet_model_epoch_{}'.format(epoch))
            torch.save(net, save_name)
            print(save_name + ' saved!')


if __name__ == "__main__":
    torch.cuda.set_device(0)
    train()
    # train_dataset = MyDataset(images_path=img_path, transform=transform, labels_path=lab_path)
    # batch_size = 8
    # dataLoader = DataLoader(train_dataset, batch_size=8, shuffle=False)
    # for data,label in dataLoader:
    #     data = data.permute(0,2,3,1).cpu()
    #     label = label.permute(0,2,3,1).cpu()
    #     labs = np.zeros((label.shape[0],label.shape[1],label.shape[2],3))
    #     labs[np.where(np.argmax(label,3)==1)] = np.array([1,0,0])
    #     imgs = np.concatenate((data,labs), axis=0)
    #     show_images(imgs, num_rows=2, num_cols=8, scale=8)
    #
    #     a=1
    print("Over !")
