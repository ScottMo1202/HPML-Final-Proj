import torch
import torchvision
import torch.nn
from torchvision import transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import argparse
from models.Alexnet4 import alexnet
from train import train, test

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", default="sgd", type=str, help="type of optimizer to use for training")
    args = parser.parse_args()

    # Create transforms
    train_transform_4 = transforms.Compose([transforms.Resize((240, 240)),
                                        transforms.RandomRotation(degrees=45),
                                        transforms.RandomVerticalFlip(p=0.5),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    test_transform_4 = transforms.Compose([transforms.Resize((240, 240)),
                                       transforms.RandomRotation(degrees=45),
                                       transforms.RandomVerticalFlip(p=0.5),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    
    # Load training and testing data
    train_images_4 = ImageFolder(root= "./data/train", transform=train_transform_4)
    test_images_4 = ImageFolder(root= "./data/test", transform=test_transform_4)
    
    # Create data loader
    trainDataLoader_4 = torch.utils.data.DataLoader(train_images_4,batch_size=128,num_workers=2, shuffle=True)
    testDataLoader_4 = torch.utils.data.DataLoader(test_images_4,batch_size=100,num_workers=2, shuffle=False)

    # Set cuda
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model, loss function and optimizers
    model = alexnet().to(device)
    Loss = torch.nn.CrossEntropyLoss()
    if args.optimizer == "sgd" or "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    elif args.optimizer == "Adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=0.1, weight_decay=5e-4)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)
    
    # Start training
    train_loss_history = []
    test_loss_history = []
    train_accuracy_history = []
    test_accuracy_history = []

    for epoch in range(50):
        train_loss = 0.0
        test_loss = 0.0
        test_total, test_correct,  test_accuracy = 0, 0, 0.0
        train_total, train_correct, train_accuracy = 0, 0, 0.0 
        train_loss, train_accuracy = train(trainDataLoader_4, optimizer, Loss, device, model)                             
        test_loss, test_accuracy = test(testDataLoader_4, Loss, device, model)
    
        # train_loss = train_loss / len(trainDataLoader_4)
        # test_loss = test_loss / len(testDataLoader_4)
        # train_accuracy = train_correct.item() / train_total
        # test_accuracy = test_correct.item() / test_total

        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        train_accuracy_history.append(train_accuracy)
        test_accuracy_history.append(test_accuracy)

        print('Epoch: %s, Train loss: %s, Train accuracy: %s, Test loss: %s, Test accuracy: %s' % (epoch, train_loss, train_accuracy, test_loss, test_accuracy))
    
    # Print Top-1 accuracies
    print("Top-1 train accuracy: ", max(train_accuracy_history))
    print("Top-1 test accuracy: ", max(test_accuracy_history))

    # Plot train and test losses
    plt.figure(1)
    plt.plot(range(50),train_loss_history,'-',linewidth=3,label='Train Loss')
    plt.plot(range(50),test_loss_history,'-',linewidth=3,label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig("./model_plots/aug4_%s_loss.png" % args.optimizer)

    # Plot train and test accuracies
    plt.figure(2)
    plt.plot(range(50), train_accuracy_history, '-', linewidth=3, label='Train Accuracy')
    plt.plot(range(50), test_accuracy_history, '-', linewidth=3, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Percentage')
    plt.grid(True)
    plt.legend()
    plt.savefig("./model_plots/aug4_%s_accuracy.png" % args.optimizer)

if __name__ == "__main__":
    main()