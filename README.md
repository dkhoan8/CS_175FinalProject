
##Step 1: Download and preprocess the SVHN data

`python Data_harvest.py`

##Step 2: Train your own models

Train single digit classifier first since it will be used for later as checkpoint.

`python single_digit_trainer.py`

This should generate a tensorflow checkpoint file:

`classifier.ckpt`

Next train the multi-digit reader

`python multi_digit_trainer.py`

This should generate a tensorflow checkpoint file:

`regression.ckpt`

## Usage

FOR Jupyter NOTEBOOK: Open project.ipynb and run all!