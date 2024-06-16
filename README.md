<p align="center">
 <h1 align="center">Speech recognition on STM32 by using STMCube-Ai</h1>
</p>

## Introduction

Here is my python source code for speech recognition - a neural network model is deployed on STM32. with my code, you could: 
* **Extract audio features and train the model**
* **Optimize the model on STM32 hardware using the STM-Cube-Ai library**

## Extraction audio
I use Google's simple audio suite, you can access it via the following link: **'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'**. I use the **MFCC** transform to be able to extract audio features to train the model:
<p align="center">
  <img src="demo/quickdraw.gif" width=600><br/>
  <i>Camera app demo</i>
</p>

## Drawing app
The script and demo will be released soon

## Dataset
The dataset used for training my model could be found at [Quick Draw dataset] https://console.cloud.google.com/storage/browser/quickdraw_dataset/sketchrnn. Here I only picked up 20 files for 20 categories

## Categories:
The table below shows 20 categories my model used:

|    Aspect                |    Edge Servers                              |      Edge Devices                              |
|--------------------------|:--------------------------------------------:|:----------------------------------------------:|
|   Resources              |   High CPU/GPU/TPU, high memory and storage  |   Limited CPU/GPU, low memory and storage      |   
|   Operating System       |    Full OS(Linux, Windows)                   |   Embedded OS, RTOS, or bare-metal             | 
|   Deployment Method      |  Containers, microservices                   |  Firmware, direct software deployment          |  
|   Model Optimization     |   Less constrained, can use larger models    | Highly optimized for size and efficiency       |  
|   Hardware Acceleration  |   GPUs, TPUs                                 | Specialized low-power accelerators             |  
|   Management             |   Easier, centralized tools                  | Challenging, may use OTA updates               |  
|   Connectivity           |   Reliable, high-bandwidth                   | Limited, may rely on intermittent connectivity |  
|   Use Case Examples      |   Industrial IoT, CDNs, smart cities         | Wearables, smart home, autonomous systems      |  

## Trained models

You could find my trained model at **trained_models/whole_model_quickdraw**

## Training

You need to download npz files corresponding to 20 classes my model used and store them in folder **data**. If you want to train your model with different list of categories, you only need to change the constant **CLASSES** at **src/config.py** and download necessary npz files. Then you could simply run **python3 train.py**

## Experiments:

For each class, I take the first 10000 images, and then split them to training and test sets with ratio 8:2. The training/test loss/accuracy curves for the experiment are shown below:

<img src="demo/loss_accuracy_curves.png" width="800"> 

## Requirements

* **python 3.6**
* **cv2**
* **pytorch** 
* **numpy**
