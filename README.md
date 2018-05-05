
# 2018 Data Science Bowl

### Installation
    
    #anaconda
    #install anaconda
    #pytorch
    $conda install pytorch torchvision -c pytorch
    $pip install -r installation.txt

### Download Kaggle dataset
    
    # loader dataset 
    kaggle competitions download -c data-science-bowl-2018    
    # relabel
    $ git clone https://github.com/lopuhin/kaggle-dsbowl-2018-dataset-fixes.git

    #external dataset
    $https://nucleisegmentationbenchmark.weebly.com/

### Visualize result with Visdom

We now support Visdom for real-time loss visualization during training!

To use Visdom in the browser:

    # First install Python server and client 
    pip install visdom
    # Start the server (probably in a screen or tmux)
    python -m visdom.server -env_path runs/visdom/
    # http://localhost:8097/

### How use
#### Step 1: Create dataset

    #(1) kaggle dataset
    ./run_createdataset.sh 
    #(2) external dataset
    ./run_createdataset_nuclei.sh

#### Step 2: Train

    ./run_train.sh
    
#### Step 3: Submission

    ./run_submission.sh


