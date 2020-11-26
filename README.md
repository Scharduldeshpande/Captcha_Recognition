# Captcha_Recognition
Automatic Captcha Recognition using CNNs

# Installation and Requirements:
Anaconda3 is recommended as it already includes many common packages

# Training
clone this repo to $Capta_recogniion_root

To download the dataset:  
!curl -LO https://github.com/AakashKumarNain/CaptchaCracker/raw/master/captcha_images_v2.zip
!unzip -qq captcha_images_v2.zip

Currently works on captcha images with 5 characters. 

training data is in a folder "training_data" in "data".
each training image is named with its label

run train.py to train
training settings could be changed in the config file


# Model Performance

After each epoch, training performance will be saved in the reults folder
