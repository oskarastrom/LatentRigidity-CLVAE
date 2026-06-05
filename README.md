# Code implementation of the article "_Latent Rigidity Regularization for Conditional VAEs in Anomaly Detection_"
### By Oskar Åström and Alexandros Sopasakis


# Install
1. Clone this github repository
```bash
git clone https://github.com/oskarastrom/LatentRigidity-CLVAE.git
cd LatentRigidity-CLVAE
```
2. Install dependencies using
```bash
pip install -r requirements.txt
```
3. (optional) Install torch and torchvision with cuda to enable GPU training 
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```


# Run CLVAE Results
1. Modify the ```train_batch.py``` file to run the desired datasets, splits, and potential other variables for the CLVAE model. This script runs multiple iterations of the training process over multiple sets of parameters.
2. Run the ```train_batch.py``` script.
```bash
python train_batch.py
```


# Implementation of CVAECapOSR

In order to incorporate Latent Rigidity into the CVAECapOSR model, modified files are found in the folder "CVAECapOSR". 

1. Follow the installation instructions at https://github.com/guglielmocamporese/cvaecaposr
```bash
# Clone the repo
$ git clone https://github.com/guglielmocamporese/cvaecaposr.git
# Go to the project directory
$ cd cvaecaposr
# Install the conda env
$ conda env create --file environment.yaml
# Activate the conda env
$ conda activate cvaecaposr
```
2. Transfer the updated files in the CVAECapOSR folder of this repository according to the following file structure
```
cvaecaposr
├── config.py (replace)
├── main.py (replace)
├── utils.py (replace)
├── main_batch.py (add)
├── models
│   ├── cvaecaposr.py (replace)
├── scripts
│   ├── batch_all_train.sh (add)
│   ├── batch_all_test.sh (add)
```
3. Run the training for all datasets, splits, and ridigity degrees using the bash script
```bash
./scripts/batch_all_train.py
```
4. Evaluate the resulting models using
```bash
./scripts/batch_all_test.py
```