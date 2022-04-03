import os
import time
import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import TestDataset, MyDataset


def Predict(img_path, labels_path, checkpoint_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    transform = transforms.Compose([transforms.ToTensor(), ])
    test_dataset = MyDataset(img_path, labels_path, transform=transform)
    batch_size = 1
    testLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    trained_model = torch.load(checkpoint_path, map_location='cuda:0')
    # trained_model = torch.load(checkpoint_path)
    trained_model = trained_model.to(device)
    accuracy = 0
    accuracylist=[]
    processlist = []
    ioulist = []
    i = 1
    for img, label in testLoader:
        start = time.clock()
        img = img.to(device)
        with torch.no_grad():
            output = torch.sigmoid((trained_model(img)))

        output_np = output.cpu().data.numpy().copy()
        output_np = np.argmax(output_np, axis=1)

        pred = np.zeros((output.shape[0], output.shape[2], output.shape[3], 3))
        pred[np.where(output_np == 1)] = np.array([0.0, 1.0, 0.0])
        end = time.clock()
        print("picture{} Process Time:{} ".format(i,end - start) )
        processlist.append(end - start)
        gt = np.argmax(label.cpu().numpy(), axis=1)
        a = np.sum(gt == output_np)
        b = gt.shape[0] * gt.shape[1] * gt.shape[2]
        accuracy = a / b
        accuracylist.append(accuracy)
        print("accuracy:", accuracy)
        iou = Iou(output_np, gt, 1)
        ioulist.append((iou))
        img = img.permute(0, 2, 3, 1).cpu().data.numpy().copy()
        label = label.permute(0, 2, 3, 1).cpu()
        labs = np.zeros((label.shape[0], label.shape[1], label.shape[2], 3))
        labs[np.where(np.argmax(label, 3) == 1)] = np.array([1, 0, 0])

        mix_img = img.copy()
        mix_img[np.where(output_np == 1)] = np.array([0.8, 0.1, 0.0])
        imgs = np.concatenate((img, labs, pred, mix_img), axis=0)

        i += 1
    print('mean_accuracy：{}'.format(np.mean(accuracylist)))
    print('mean_processtime：{}'.format(np.mean(processlist)))
    print('mean_iou：{}'.format(np.mean(ioulist)))

def Iou(input, target, classes=1):
    intersection = np.logical_and(target == classes, input == classes)
    union = np.logical_or(target == classes, input == classes)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def compute_f1(prediction, target):
    prediction.tolist(), target.tolist()
    img, target = np.array(prediction).flatten(), np.array(target).flatten()
    f1 = f1_score(y_true=target, y_pred=img)
    return  f1


if __name__=="__main__":
    data_path = "./dataset/test/"
    img_path = os.path.join(data_path, 'images')
    labels_path = os.path.join(data_path, 'labels')
    checkpoint_path = './checkpoints/2022-04-03_15_56_32/'
    Predict(img_path, labels_path, checkpoint_path)
