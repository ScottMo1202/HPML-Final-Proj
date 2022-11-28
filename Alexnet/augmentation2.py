import torch
import torchvision
import torch.nn
from torchvision import transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import argparse
from models.Alexnet2 import alexnet
from train import train, test

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", default="sgd", type=str, help="type of optimizer to use for training")
    args = parser.parse_args()

    # Create transforms
    train_transform_2 = transforms.Compose([transforms.Resize((240, 240)),
                                        transforms.CenterCrop(size=(224, 224)),
                                        transforms.Grayscale(num_output_channels=1),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914), (0.2023))])
    test_transform_2 = transforms.Compose([transforms.Resize((240, 240)),
                                       transforms.CenterCrop(size=(224, 224)),
                                       transforms.Grayscale(num_output_channels=1),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.4914), (0.2023))])
    
    # Load training and testing data
    train_images_2 = ImageFolder(root= "./data/train", transform=train_transform_2)
    test_images_2 = ImageFolder(root= "./data/test", transform=test_transform_2)

    # Create data loader
    trainDataLoader_2 = torch.utils.data.DataLoader(train_images_2,batch_size=128,num_workers=2, shuffle=True)
    testDataLoader_2 = torch.utils.data.DataLoader(test_images_2,batch_size=100,num_workers=2, shuffle=False)

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
        train_loss, train_accuracy = train(trainDataLoader_2, optimizer, Loss, device, model)                             
        test_loss, test_accuracy = test(testDataLoader_2, Loss, device, model)
    
        # train_loss = train_loss / len(trainDataLoader_2)
        # test_loss = test_loss / len(testDataLoader_2)
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
    plt.savefig("./model_plots/aug2_%s_loss.png" % args.optimizer)

    # Plot train and test accuracies
    plt.figure(2)
    plt.plot(range(50), train_accuracy_history, '-', linewidth=3, label='Train Accuracy')
    plt.plot(range(50), test_accuracy_history, '-', linewidth=3, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Percentage')
    plt.grid(True)
    plt.legend()
    plt.savefig("./model_plots/aug2_%s_accuracy.png" % args.optimizer)

if __name__ == "__main__":
    main()