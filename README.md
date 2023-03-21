# SBMx

Alleviating Popularity Bias in Session-based Recommendation Considering Long-tail Distribution

### **Overall Framework of SBMx**
<img src=./assets/SBMx_framework.jpg>


## Setups
[![Python](https://img.shields.io/badge/python-3.9.15-blue?logo=python&logoColor=FED643)](https://www.python.org/downloads/release/python-3915/)
[![Pytorch](https://img.shields.io/badge/pytorch-1.13.0-red?logo=pytorch)](https://pytorch.org/get-started/previous-versions/)


## Datasets
The dataset name must be specified in the `--dataset` argument
- [Yoochoose 1/64](https://www.kaggle.com/chadgostopp/recsys-challenge-2015) (using latest 1/64 fraction due to the amount of full dataset) <br>
- [Diginetica](https://competitions.codalab.org/competitions/11161 #learn_the_details-data2)
- [RetailRocket](https://www.kaggle.com/retailrocket/ecommerce-dataset)

After downloaded the datasets, you can put them in the folder `Datasets/` and preprocess datasets by running `Datasets/preprocess_code/{dataset_name}.ipynb` below line. 


## Train and Test
```
python main.py \
    --dataset diginetica \
    --batchSize 128 \
    --hiddenSize 100 \
    --epoch 30 \
    --lr 0.001 \
    --lr_dc 0.1 \
    --lr_dc_step 3 \
    --l2 1e-5 \
    --step 1 \
    --mixup_lam 0.9 \
    --mixup_pct 0.5 \
```


## Citation
Please cite our paper if you use our code:
```
Heeyoon Yang, Jee-Hyong Lee.(2022).
Alleviating Popularity Bias in Session-based Recommendation Considering Long-tail Distribution.
한국정보과학회 학술발표논문집,(),532-534.
```
