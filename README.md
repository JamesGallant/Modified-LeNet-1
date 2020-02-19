# Description
This is a slightly modified version of LeNet which has been trained on the CIFAR-10 dataset. The model has been trained for 40 epochs on a laptop gpu and has a accuracy of 74%. Because this is sort of a demo but still a useable model I have not written the most optimal python code or use case (In case anyone finds their way here). Before I knew about adaptive learning I have implemented my own version here. In short the learning rate will change depending on the accuracy of every epoch.
# Using the models
Install the dependencies the requirements.txt file.
```
pip install -r requirements.txt
```
This should work fine to run this on the cpu, if you would like to run this on the gpu first install <a href=https://developer.nvidia.com/cuda-downloads target="_blank"> cuda </a> from nvidia and I would suggest to download pytorch from the pytorch website to ensure you have gpu support. 

The image classes by index of CIFAR-10 is:
```
0: plane
1: car
2: bird
3: cat
4: deer
5: dog
6: frog
7: horse
8: ship
9: truck
```
In the folder named CIFAR_hires is a sample of an image that I randomly got from the internet to check if the model works. This is where the usage becomes bulky...

Open the file and rename the imageName variable to one of the classes listed above and leave the extension, eg. plane.jpg, or horse.jpg and call the script. To retrain this model change the TRAIN and DynamicLearning value to True.

  ```
  python simpleNet.py
  ```
 # How the adaptive learning rate works in this model
 The accuracy of each epoch is stored in a que, as the new accuracies come in the que is updated to always contain two values, the current accuracy and the previous accuracy. The learning rate is updated accordingly by either decreasing or increasing this rate depending on the accuracy. If the accuracies are the same the model will "vote" by generating a array of ten binary digit randomly. If the sum is greater than five the learning rate will increase, if less than five the learning rate will decrease and if it is the same the learning rate will reset to default and the model will start learning again. As we can see below it takes less epochs to get to the same accuracy without, which I thought was pretty cool. Next would be to increase the complexity of the network. On the same number of epochs 80% can be achieved.
 
 ![Image of accuracy](https://github.com/JamesGallant/Modified-LeNet-1/blob/master/images/CIFAR10%20accuracy.png)

  
