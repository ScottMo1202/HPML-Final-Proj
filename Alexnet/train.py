import torch

def train(trainDataLoader, optimizer, Loss, device, model):
    train_loss = 0.0
    train_total, train_correct, train_accuracy = 0, 0, 0.0
    train_loss_array = []
    train_accuracy_array = []

    for i, (images, labels) in enumerate(trainDataLoader):   
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        predicted_output = model(images).to(device)
        fit = Loss(predicted_output, labels).to(device)
        fit.backward()
        optimizer.step()

        _, predicted = torch.max(predicted_output, 1)
        train_total += labels.numel()
        train_correct += (predicted == labels).sum()
        train_loss += fit.item()
        train_accuracy = train_correct / train_total
        train_accuracy = train_accuracy.item()
        train_loss_array.append(train_loss)
        train_accuracy_array.append(train_accuracy)
    
    train_loss = sum(train_loss_array) / len(train_loss_array)
    train_accuracy = max(train_accuracy_array)

    return train_loss, train_accuracy

def test(testDataLoader, Loss, device, model):
    test_loss = 0.0
    test_total, test_correct, test_accuracy = 0, 0, 0.0
    test_loss_array = []
    test_accuracy_array = []

    for i, (images, labels) in enumerate(testDataLoader):
        with torch.no_grad():
            images = images.to(device)
            labels = labels.to(device)
            predicted_output = model(images)

            _, predicted = torch.max(predicted_output, 1)
            test_total += labels.numel()
            test_correct += (predicted == labels).sum()
            fit = Loss(predicted_output, labels)
            test_loss += fit.item()
            test_accuracy = test_correct / test_total
            test_accuracy = test_accuracy.item()
            test_loss_array.append(test_loss)
            test_accuracy_array.append(test_accuracy)
    
    test_loss = sum(test_loss_array) / len(test_loss_array)
    test_accuracy = max(test_accuracy_array)
    
    return test_loss, test_accuracy