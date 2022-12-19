# HPML-Final-Proj
This project trains three Deep Learning models with four data augmentation methods and three optimizers. For each model, we will determine the best pair of augmentation and optimizer, and finally we will find the model with best performance based on the results we obtained.

## Run AlexNet code in HPC
Four python files are written for each augmentations. Below are arguments that can be passed to the files:
* --optimizer: specify which optimizer (SGD, Adagrad, Adam) to use for training. Default: sgd
* --num_gpu: specify the number of GPUs to run

Here are the examples to run four Alexnet python files:
''

## Run GoogleNet and VGG code in HPC
File run-googlenet-VGG.py is the gateway to trian and test with GoogleNet and VGG16. Below are arguments that can be passed to the file:
* --optimier: specify which optimizer, sgd, adagrad, or adam, to use. Default: sgd
* --cuda: If passed, the job will be trained with GPU. 
* --aug: specify which augmentation method to use. Available options: HorizontalFlip, CenterCrop, ColorJitter, and VerticalFlip. Default: HorizontalFlip
* --batchSize: specify the batch size to load training data. Default: 32
* --numGPUs: specify the number of GPUs to use to train and test the job. Default: 1
* --model: specify the deep learning model to use. Available options: googlenet and vgg. Default: googlenet.

Here is and example of running run-googlenet-VGG.py: <br>
`python3 run-googlenet-VGG.py --cuda --aug=ColorJitter --model=vgg --numGPUs=2 --batchSize=32 --optimizer=sgd`<br>
It means using VGG16 network with Color Jitter augmentation and SGD optimizer to train and test. The job will be run with training batch size 32 and on 2 GPUs. After training and testing, it will plot the training and testing accuracies/losses and save them in two pngs. For example, HorizontalFlip__train_test_accuracies_sgd_2_vgg.png means the accuracy plot using VGG16 network with Horizontal Flip and SGD on 2 GPUs.

## GoogLeNet Results
### Results – GoogLeNet with SGD
HorizontalFlip train/test accuracy and loss:<br>
![alt text](./googlenet-results/SGD/HF_train_test_accuracies_SGD.png)
![alt text](./googlenet-results/SGD/HF_train_test_losses_SGD.png)
    
CenterCrop & Grayscale train/test accuracy and loss:<br>
![alt text](./googlenet-results/SGD/CC_train_test_accuracies_SGD.png)
![alt text](./googlenet-results/SGD/CC_train_test_losses_SGD.png)

ColorJitter & Pad train/test accuracy and loss:<br>
![alt text](./googlenet-results/SGD/CJ_train_test_accuracies_SGD.png)
![alt text](./googlenet-results/SGD/CJ_train_test_losses_SGD.png)

    
Random rotation & VerticalFlip train/test accuracy and loss:<br>
![alt text](./googlenet-results/SGD/VF_train_test_accuracies_SGD.png)
![alt text](./googlenet-results/SGD/VF_train_test_losses_SGD.png)
    
Augmentation CenterCrop & Grayscale has the best performance of GoogLeNet with SGD.

### Results – GoogLeNet with Adagrad
HorizontalFlip train/test accuracy and loss:<br>
![alt text](./googlenet-results/Adagrad/HF_train_test_accuracies_Adagrad.png)
![alt text](./googlenet-results/Adagrad/HF_train_test_losses_Adagrad.png)
   
CenterCrop & Grayscale train/test accuracy and loss:<br>
![alt text](./googlenet-results/Adagrad/CC_train_test_accuracies_Adagrad.png)
![alt text](./googlenet-results/Adagrad/CC_train_test_losses_Adagrad.png)
   
ColorJitter & Pad train/test accuracy and loss:<br>
![alt text](./googlenet-results/Adagrad/CJ_train_test_accuracies_Adagrad.png)
![alt text](./googlenet-results/Adagrad/CJ_train_test_losses_Adagrad.png)
   
Random rotation & VerticalFlip train/test accuracy and loss:<br>
![alt text](./googlenet-results/Adagrad/VF_train_test_accuracies_Adagrad.png)
![alt text](./googlenet-results/Adagrad/VF_train_test_losses_Adagrad.png)
   
Augmentation HorizontalFlip has the best performance of GoogLeNet with Adagrad.



### Results – GoogLeNet with Adam
HorizontalFlip train/test accuracy and loss:<br>
![alt text](./googlenet-results/Adam/HF_train_test_accuracies_Adam.png)
![alt text](./googlenet-results/Adam/HF_train_test_losses_Adam.png)
   
CenterCrop & Grayscale train/test accuracy and loss:<br>
![alt text](./googlenet-results/Adam/CC_train_test_accuracies_Adam.png)
![alt text](./googlenet-results/Adam/CC_train_test_losses_Adam.png)
   
ColorJitter & Pad train/test accuracy and loss:<br>
![alt text](./googlenet-results/Adam/CJ_train_test_accuracies_Adam.png)
![alt text](./googlenet-results/Adam/CJ_train_test_losses_Adam.png)
   
Random rotation & VerticalFlip train/test accuracy and loss:<br>
![alt text](./googlenet-results/Adam/VF_train_test_accuracies_Adam.png)
![alt text](./googlenet-results/Adam/VF_train_test_losses_Adam.png)
   
Augmentation CenterCrop & Grayscale has the best performance of GoogLeNet with Adam.


### Results – Alexnet with SGD
HorizontalFlip train/test accuracy and loss:






