
# CS175 ML Final Project

## File Descriptions:

1. Data_harvest.py: download data from given url in the script (svhn from standford).
2. digit_struct.py: transforms raw dataset into a structural table with cols as labels and rows as data.
3. Model.py: sets up model of the CNN with initialized random biases and weights.
4. Single_digit_trainer.py: Train and create a model with cropped images that only contain 1 digit.
5. Multi_digit_trainer.py: Train and create another model with cropped images that only contain more than 1 digit and less than 6 digits.
6. project.ipynb: contains codes that can test the models with raw data.
7. project.html: a printed version of project.ipynb with results and code.

## How to train:
### Step 1: Navigate to src directory

### Step 2: Download data with Data_harvest.py. Run command:
    python Data_harvest.py

    -- This will produce a directory called 'data' --
    
### Step 3: Train single digit recognizer with single_digit_trainer.py. Run command:
    python single_digit_trainer.py

    -- This will produce a model .ckpt file (classifier.ckpt) --

### Step 3: Train multi digits recognizer with multi_digit_trainer.py. Run command:
    python multi_digit_trainer.py.py

    -- This will produce a model .ckpt file (regression.ckpt) --

### Step 4: Test the models with your notebook environment, Jupyter Notebook recommended:
    Open project.ipynb with jupyter notebook and run.