# Reading Comprehension with Neural Networks

**WARNING: This code only works with Python 3.6!**
You can check your version of Python with `python --version`.

## Table of Contents

- [Installation](#installation)
- [Downloading the data](#downloading-the-data)

## Installation

To run the code in this project, you need to use Python 3.6. This is because we depend 
on [`AllenNLP`](http://allennlp.org/), which only works with Python 3.6

[Conda](https://conda.io/) will set up a virtual environment with the exact version of Python
used for development along with all the dependencies needed to run the code.

1.  [Download and install Conda](https://conda.io/docs/download.html).

2.  Change your directory to your clone of this repo.

    ```
    cd CSE447_RNN
    ```

3.  Create a Conda environment with Python 3.

    ```
    conda create -n 447_rnn python=3.6
    ```

4.  Now activate the Conda environment.
    You will need to activate the Conda environment in each terminal in which you 
    want to run code from this repo.

    ```
    source activate 447_rnn
    ```

5.  Install the required dependencies.

    ```
    pip install -r requirements.txt
    ```
    
6.  Install the SpaCy model.

    ```
    python -m spacy download en
    ```

7. Visit [http://pytorch.org](http://pytorch.org) and install the relevant PyTorch 1.4 package (latest stable version).


You should now be able to test your installation with `pytest -v`.  Congratulations!

## Downloading the data

By default, the code expects SQuAD data in a folder at `./squad/`, with files 
`train_small.json`, `val_small.json`, and `test_small.json`.

In addition, the code expects `glove.6B.50d` vectors in `./glove/`. You can download
these vectors from the [GloVe website @ StanfordNLP](https://nlp.stanford.edu/projects/glove/) ---
`glove.6B.zip` is the archive you want, and unzipping it will give you the vectors. Feel free to
experiment with using the other GloVe vectors as well!

## Running Experiments
To train a model (with GPU) and save the parameters, run:
```
python train_model.py --save-dir saved_models/<model_name> --cuda --num-epochs 25
```
To test the mode, run:
```
python train_model.py --load-path saved_models/<model_name>/<desired .pth file>
```
## Monitoring with Tensorboard
Run 
```
tensorboard --log-dir saved_models/ --host 127.0.0.1
```
You can access `tensorboard` at http://localhost:6006.

Assuming `tensorboard` is running on a remote machine on port 6006, you can access tensorboard on port 16006 on your local machine by running the following on your local machine:
```
ssh -NL 16006:127.0.0.1:6006 <user>@<server ip or hostname>
```

## Demo
Run
```
python train_model.py --load-path <path to .pth file> --demo
```
and access the local demo server at http://localhost:5000. For access from a remote server, refer to the methods described in the above section.

[![Image from Gyazo](https://i.gyazo.com/ec9b225759d10f5d5abbaeac23b75139.gif)](https://gyazo.com/ec9b225759d10f5d5abbaeac23b75139)

