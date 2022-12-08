# HPML-Final-Proj

## Run AlexNet code in HPC
A sample sbatch file is given for each AlexNet model code in the folder named "Alexnet". Simply changing the string after "--optimizer" can make the program running with 
different optimizers. Four model code programs are written for AlexNet and each of which uses different data augmentation techniques.

## Run GoogleNet and VGG code in HPC
File run-googlenet-VGG.py is the gateway to trian and test with GoogleNet and VGG16. Below are arguments that can be passes to the file:
* --optimier: specify which optimizer, sgd, adagrad, or adam, to use. Default: sgd
* --cuda: If passed, the job will be trained with GPU. 
* --aug: specify which augmentation method to use. Available options: HorizontalFlip, CenterCrop, ColorJitter, and VerticalFlip. Default: HorizontalFlip
* --batchSize: specify the batch size to load training data. Default: 32
* --numGPUs: specify the number of GPUs to use to train and test the job. Default: 1
* --model: specify the deep learning model to use. Available options: googlenet and vgg. Default: googlenet.

Here is and example of running run-googlenet-VGG.py: <br>
`python3 run-googlenet-VGG.py --cuda --aug=ColorJitter --model=vgg --numGPUs=2 --batchSize=32 --optimizer=sgd`<br>
It means using VGG16 network with Color Jitter augmentation and SGD optimizer to train and test. The job will be run with training batch size 32 and on 2 GPUs.

