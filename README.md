## Bert based sentiment classification using IMDB50K movie review dataset


### Data 

Download and place the data file (in csv) format inside `input` folder. The data can be downloaded from [here](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)


### Create conda environment

All required libraries are listed inside `environment.yml` file. You can create an exact environment using the following command:

`conda env create -f environment.yml`

### Training

To train the model, run `python3 train.py` 

To change the hyperparameters, please look into `config.py`

### Deploying the model using Flask backend

Coming soon!


## Reference

[1] https://github.com/abhishekkrthakur/bert-sentiment