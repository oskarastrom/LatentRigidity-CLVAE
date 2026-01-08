# Code implementation of the article "_Latent Rigidity Regularization for Conditional VAEs in Anomaly Detection_"
## By Oskar Åström and Alexandros Sopasakis


### Install
1. Clone this github repository
```bash
git clone https://github.com/oskarastrom/LatentRigidity-CLVAE.git
cd LatentRigidity-CLVAE
```

3. Install dependencies using
```bash
pip install -r requirements.txt
```
3. (optional) Install torch and torchvision with cuda to enable GPU training 
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```


### Run
1. Modify the ```train_batch.py``` file to run the desired datasets, splits, and potential other variables. This script runs multiple iterations of the training process over multiple sets of parameters.
2. Run the ```train_batch.py``` script.
```bash
python train_batch.py
```
