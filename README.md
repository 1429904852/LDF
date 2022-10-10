# LDF

Code and data for "Label-Driven Denoising Framework for Multi-Label Few-Shot Aspect Category Detection" (Findings of EMNLP 2022)

## Overview

<img src="figs/EMNLP2022_LDF.png" style="width:200px height:300px" />

- In this paper, we propose a Label-Driven Denoising Framework (LDF) to alleviate the noise problems for the FS-ACD task.
- Label-Driven Denoising Framework contains a label-guided attention strategy to filter noisy words and generate a representative prototype for each aspect, and a label-weighted contrastive loss to avoid generating similar prototypes for
semantically-close aspect categories.

## Setup

### Requirements
```bash
+ python 3.7
+ tensorflow 2.4.0
+ keras 2.4.3
+ sklearn 0.0
+ numpy 1.19.5
```

### Download word embedding
please download the glove.6B.50d embedding ([Link](https://drive.google.com/file/d/1vCm_X2vrSSwLICwmm4NW2-dXfNtV8TFg/view?usp=sharing)) and put it into word_embedding folder

### Model configuration

- you can choose one or multiple methods at one time in the model_list
```bash
e.g., model_list = [None, 'AWATT+LAS', 'LDF_AWATT']

# code:             corresponding model:
#  None             the original AWATT model
# 'AWATT_LAS'       AWATT+LAS
# 'AWATT_LCL'       AWATT+LCL
# 'AWATT_SCL'       AWATT+SCL
# 'LDF_AWATT'       LDF-AWATT
# 'HATT'            the original HATT model
# 'HATT_LAS'        HATT+LAS
# 'HATT_LCL'        HATT+LCL
# 'HATT_SCL'        HATT+SCL
# 'LDF-HATT'        LDF-HATT
```

- you can choose one or multiple datasets at one time in the dataset_list
```bash
e.g., dataset_list = ['FewAsp', 'FewAsp(single)', 'FewAsp(multi)']
```

- you can choose one or multiple configs at one time in the config_list
```bash
e.g., config_list = [[2, 5, 5], [1, 5, 10], [1, 10, 5], [1, 10, 10]]

# [2, 5, 5] stands for: two(2) '5'-way-'5'-shot meta-tasks for two batch-size
# [1, 5, 10] stands for: one(1) '5'-way-'10'-shot meta-task for one batch-size
# [1, 10, 5] stands for: one(1) '10'-way-'5'-shot meta-task for one batch-size
# [1, 10, 10] stands for: one(1) '10'-way-'10'-shot meta-task for one batch-size
```

## Usage

- You can use the folowing command to train and test LDF on the FS-ACD task:

```bash
python train_and_test.py
```

- The final results can be saved in the excel file you specified:

```bash
e.g., pd.DataFrame(result_list).to_excel("/data1/zhaof/LDF/" + 'result.xlsx')

```

## Citation

If the code is used in your research, please cite our paper.



