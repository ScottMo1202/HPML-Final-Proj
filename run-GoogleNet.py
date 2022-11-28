import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn
import time
import sys
import argparse
import torch.nn.functional as F 
from googlenet import GoogleNet
from random import shuffle
import matplotlib.pyplot as plt
# https://github.com/kuangliu/pytorch-cifar
# https://github.com/weiaicunzai/pytorch-cifar100/blob/master/train.py
import os
import shutil
# https://github.com/DivyaMunot/Face-mask-detection-with-tensorflow

def train(train_loader, device, optimizer, network, criterion, epoch, train_test_stats):
    print("Training Epoch: " + str(epoch + 1))
    total_time_start = time.monotonic_ns()
    top_accuracy = 0.0
    train_loss = 0
    correct = 0
    total = 0
    total_data_load_time = 0
    total_train_time = 0
    total_transfer_time = 0
    network.train()
    data_load_time_start = time.monotonic_ns()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        data_load_time_end = time.monotonic_ns()
        inputs, targets = inputs.to(device), targets.to(device)
        data_transfer_time_end = time.monotonic_ns()
        total_data_load_time += data_load_time_end - data_load_time_start
        total_transfer_time += data_transfer_time_end - data_load_time_end
        # The training part will be commented out in C3
        train_start = time.monotonic_ns()
        optimizer.zero_grad()
        outputs = network(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += float(loss.item())
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        train_end = time.monotonic_ns()
        total_train_time += train_end - train_start
        data_load_time_start = time.monotonic_ns()
        the_accuracy = 100.0 * (correct / total)
        if the_accuracy > top_accuracy:
            top_accuracy = the_accuracy
    total_time_end = time.monotonic_ns()
    # print("The train loss is " + str(train_loss))
    # print("The data loading time for epoch " + str((epoch + 1)) + " is " + str((total_data_load_time / 1000000000)) + " seconds.")
    # print("The train time for epoch " + str((epoch + 1)) + " is " + str(((total_train_time + total_transfer_time) / 1000000000)) + " seconds.")
    # print("The total time for epoch " + str((epoch + 1)) + " is " + str(((total_time_end - total_time_start) / 1000000000)) + " seconds.")
    print("The best accuracy for epoch " + str(epoch + 1) + " is " + str(top_accuracy) + "%.")
    train_test_stats["train_times"].append((total_train_time + total_transfer_time) / 1000000000)
    train_test_stats["train_losses"].append(train_loss)
    train_test_stats["top_train_accuracies"].append(top_accuracy)
    return train_loss, total_time_end - total_time_start, top_accuracy

def test(test_loader, device, network, criterion, epoch, train_test_stats):
    print("Testing Epoch: " + str(epoch + 1))
    total_time_start = time.monotonic_ns()
    network.eval()
    test_loss = 0
    correct = 0
    total = 0
    total_test_time = 0
    total_transfer_time = 0
    top_accuracy = 0.0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_transfer_time_start = time.monotonic_ns()
            inputs, targets = inputs.to(device), targets.to(device)
            data_transfer_time_end = time.monotonic_ns()
            total_transfer_time += data_transfer_time_end - data_transfer_time_start
            test_start = time.monotonic_ns()
            outputs = network(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            test_end = time.monotonic_ns()
            total_test_time += test_end - test_start
            the_accuracy = 100.0 * (correct / total)
            if the_accuracy > top_accuracy:
                top_accuracy = the_accuracy

        total_time_end = time.monotonic_ns()
        train_test_stats["test_times"].append((total_test_time + total_transfer_time) / 1000000000)
        train_test_stats["test_losses"].append(test_loss)
        train_test_stats["top_test_accuracies"].append(top_accuracy)
        return test_loss, total_time_end - total_time_start, top_accuracy

def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    all_images = os.listdir(SOURCE)
    print(len(all_images))
    # shuffle(all_images)
    # splitting_index = round(SPLIT_SIZE * len(all_images))
    # train_images = all_images[:splitting_index]
    # test_images = all_images[splitting_index:]

    # for img in train_images:
    #     src = os.path.join(SOURCE, img)
    #     dst = os.path.join(TRAINING, img)
    #     if os.path.getsize(src) <= 0:
    #         print(img + " is zero length, so ignoring!")
    #     else:
    #         shutil.copyfile(src, dst)
        
    # for img in test_images:
    #     src = os.path.join(SOURCE, img)
    #     dst = os.path.join(TESTING, img)
    #     if os.path.getsize(src) <= 0:
    #         print(img + " is zero length, so ignoring!")
    #     else:
    #         shutil.copyfile(src, dst)

def main():

    parser = argparse.ArgumentParser(description='Face Mask Detection Project')
    parser.add_argument('--optimizer', default='sgd', type=str, help='the optimizer to use')
    parser.add_argument('--cuda', action="store_true",  help='the use of GPU')
    parser.add_argument('--aug', default="HorizontalFlip", type=str, help='Specify the augumentation method.')
    parser.add_argument('--batchSize', default=128, type=int, help="Define the batch size.")
    parser.add_argument('--numGPUs', default=1, type=int, help="Define the number of GPUs to use.")
    args = parser.parse_args()

    WITH_MASK_PATH = "./data/with_mask/"
    WITHOUT_MASK_PATH = "./data/without_mask/"
    WITH_MASK_PATH_TRAINING = "./data/train/with_mask/"
    WITH_MASK_PATH_TESTING = "./data/test/with_mask/"
    WITHOUT_MASK_PATH_TRAINING = "./data/train/without_mask/"
    WITHOUT_MASK_PATH_TESTING = "./data/test/without_mask/"
    
    # try:
    #     shutil.rmtree(WITH_MASK_PATH_TRAINING)
    #     shutil.rmtree(WITH_MASK_PATH_TESTING)
    #     shutil.rmtree(WITHOUT_MASK_PATH_TRAINING)
    #     shutil.rmtree(WITHOUT_MASK_PATH_TESTING)
    #     print("Deleted existing train and test data")
    # except:
    #     print("Failed to delete training and testing directories.")
    
    # os.mkdir(WITH_MASK_PATH_TRAINING)
    # os.mkdir(WITH_MASK_PATH_TESTING)
    # os.mkdir(WITHOUT_MASK_PATH_TRAINING)
    # os.mkdir(WITHOUT_MASK_PATH_TESTING)
    the_transform = ''
    if args.aug == "HorizontalFlip":
        the_transform = transforms.Compose([
            transforms.Resize(240),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
    elif args.aug == "CenterCrop": 
        the_transform = transforms.Compose([
            transforms.Resize(240),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914), (0.2023))
        ])
    elif args.aug == "ColorJitter":
        the_transform = transforms.Compose([
            transforms.Resize(240),
            transforms.RandomCrop(220),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            transforms.Pad([8,8,8,8], fill=0, padding_mode="constant"),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
    else:
        the_transform = transforms.Compose([
            transforms.Resize(240),
            transforms.RandomCrop(220),
            transforms.RandomRotation(degrees=45),
            transforms.RandomVerticalFlip(p = 0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
    split_size = 0.9
    # split_data(WITH_MASK_PATH, WITH_MASK_PATH_TRAINING, WITH_MASK_PATH_TESTING, split_size)
    # split_data(WITHOUT_MASK_PATH, WITHOUT_MASK_PATH_TRAINING, WITHOUT_MASK_PATH_TESTING, split_size)   
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    train_set = torchvision.datasets.ImageFolder("../../data/train/", transform=the_transform)
    test_set = torchvision.datasets.ImageFolder("../..//data/test/", transform=the_transform)
    train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batchSize, num_workers=2, shuffle=True
        )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=32, num_workers=2, shuffle = True
    )
    model = GoogleNet().to(device)
    if device == 'cude':
        model = torch.nn.DataParallel(model, device_ids=list(range(args.numGPUs)))
        cudnn.benchmark = True

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=0.1,
                        momentum=0.9, weight_decay=5e-4, nesterov=False)
    elif args.optimizer == 'nesterov':
        optimizer = optim.SGD(model.parameters(), lr=0.1,
                        momentum=0.9, weight_decay=5e-4, nesterov=True)
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=0.1, weight_decay=5e-4)
    elif args.optimizer == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=0.1, weight_decay=5e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.1,weight_decay=5e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    criterion = nn.CrossEntropyLoss()
    all_time = 0
    all_train_loss = 0
    all_test_loss = 0
    best_train_accuracy = 0.0
    best_test_accuracy = 0.0
    train_test_stats = {"train_losses": [], "top_train_accuracies": [], "test_losses":[], "top_test_accuracies": [], "train_times": [], "test_times": []}
    print("Running with optimizer " + args.optimizer + " and with augmentation " + args.aug)
    for i in range(50):
        train_loss, train_epoch_time, train_accuracy = train(train_loader, device, optimizer, model, criterion, i, train_test_stats)
        test_loss, test_epoch_time, test_accuracy = test(test_loader, device, model, criterion, i, train_test_stats)
        all_time += train_epoch_time + test_epoch_time
        all_train_loss += train_loss
        all_test_loss += test_loss
        if train_accuracy > best_train_accuracy:
            best_train_accuracy = train_accuracy
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = train_accuracy
        scheduler.step()
    print("The top train accuracy is among 50 epochs is " + str(best_train_accuracy) + "%.")  
    print("The top test accuracy is among 50 epochs is " + str(best_test_accuracy) + "%.")  
    print(train_test_stats)
    
if __name__ == "__main__":
    main()