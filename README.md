# README


## Hardware

Google Colaboratory
* CPU: Intel(R) Xeon(R) CPU @ 2.30GHz
* RAM: 13298580KB
* GPU: Nvidia P100-PCIE


## OS

Ubuntu 18.04.5 LTS


## Third party software

To use GPU, following packages are required.

* CUDA 11.1.105
* cuDNN 7.6.5

Above packages are pre-installed in Google Colaboratory.
And following python packages are needed.

* fugashi 1.1.2
* ipadic 1.0.0
* PyTorch 1.11.0
* Transformers 4.20.0

To install above python packages, execute following command.

```bash
pip install -r requirements.txt
```


## Random seed

Set random seed value to 0.
For detail, see the function ```set_seed``` in  ```src/utils.py```.


## Train and predict

Training code and prediction code are separated.

```bash
python train.py
```

Training code outputs trained model to the directory ```MODEL_CHECKPOINT_DIR``` defined at ```settings.json```.
Prediction code use that trained model.

```bash
python predict.py
```


## Predict using trained model

```bash
python predict.py
```


## Prerequisites

Nothing.


## Notification

Trained model is overwritten.
