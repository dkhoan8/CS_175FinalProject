
##Step 1: Download and preprocess the SVHN data

`python svhn_data.py`

##Step 2: Train your own models

Train single digit classifier first since it will be used for later as checkpoint.

`python train_classifier.py`

This should generate a tensorflow checkpoint file:

`classifier.ckpt`

Next train the multi-digit reader

`python train_regressor.py`

This should generate a tensorflow checkpoint file:

`regression.ckpt`

## Usage

FOR LOCAL:

The single digit reader for an image file  Cropped_Test_Img/any_file_cropped.png `python single_digit_reader.py Single_Digit_Test/any_file_cropped.png`

The multi digit reader for an image file  Cropped_Test_Img/any_file_cropped.png `python mulit_digit_reader.py  Multi_Digit_Test/any_file_cropped.png`

FOR NOTEBOOK:

RUN ALL CELLS.