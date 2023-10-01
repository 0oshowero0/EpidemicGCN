# EpiGCN

## Introduction

This repo is the code for the SIGSPATIAL 23 paper: [Devil in the Landscapes: Inferring Epidemic Exposure Risks from Street View Imagery](https://doi.org/10.1145/3589132.3625596).


## System Requirement

### Example System Information
Operating System: Ubuntu 22.04 LTS

CPU: Intel(R) Xeon(R) Platinum 8358 x2

GPU: NVIDIA GeForce RTX 4090

Memory: 512G DDR4 ECC Memory



### Installation Guide
Typically, a modern computer with fast internet can complete the installation within 10 mins.

1. Download Anaconda according to [Official Website](https://www.anaconda.com/products/distribution), which can be done by the following command (newer version of anaconda should also work)
``` bash
wget -c https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
```
2. Install Anaconda through the commandline guide. Permit conda init when asked.
``` bash
./Anaconda3-2023.09-0-Linux-x86_64.sh
```
3. Quit current terminal window and open a new one. You should be able to see (base) before your command line. 

4. Use the following command to install pre-configured environment through the provided `.yml` file (you should go to the directory of this project before performing the command).
``` bash
conda env create -f ./requirements.yml
```

5. Finally, activate the installed environment. Now you can run the example code through the following chapter.
``` bash
conda activate epigcn
```

(Optional) If you need to exit the environment for other project, use the following command.

``` bash
conda deactivate 
```

## Run the code

### 0. Download & Unzip data
Download the dataset file from this link: https://cloud.tsinghua.edu.cn/f/806122f984ae40cdba81/?dl=1 and uncompress it to the project root.
``` bash
wget -c --content-disposition https://cloud.tsinghua.edu.cn/f/806122f984ae40cdba81/?dl=1 
tar -zvxf dataset.tar.gz
```


### 1. Train the CV feature extractor
Note: We cannot provide the street view images used in our project due to the restriction of Google. Therefore, the following code is only for reference. Instead, we provide the embeddings of these images for the following steps.

``` bash
python Engine_Feature_Extractor.py --data Street --cv_base_arch ResNet18 --embedding_size 512 --max_epoch 80 --early_stop 30 --init_learning_rate 3e-5 --lambda 3e-5
```

### 2. Train the GCN predictor
``` bash
python Engine_GCN.py --data Street --model EpiGCN --feature_size 512 --embedding_size 512 --max_epoch 200 --early_stop 20 --init_learning_rate 1e-5 --lambda 3e-5 --k 50 --scheduler
```

### 3. Check the results
The trained model/tensorboard records will be saved in ```results```, and the predition can be found in ```output```.
``` bash
cat ./results/epigcn_Street_EpiGCN_Emb_512_Lr_1e-05_Lamb_3e-05_beta_0.999_k_50_scheduler.csv
cat ./output/epigcn_Street_EpiGCN_Emb_512_Lr_1e-05_Lamb_3e-05_beta_0.999_k_50_scheduler_00_test_pred.csv | head -n 5
```