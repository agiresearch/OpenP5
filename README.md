# OpenP5: An Open-source Platform for Developing, Fine-tuning, and Evaluating LLM-based Recommenders

____________

# Suzumura Lab Documenation

This is a fork from the original project of OpenP5. The idea is to conduct experiments and code improvements from their code.

## Installation Steps
If you're using python virtual environment, make sure to create a venv folder with python `3.9.7` which was the version used to execute the scripts.

```
python -m venv venv
source venv/bin/activate
```

Afterwards, run the following commands to install python packages:

```
pip install -U pip
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==4.33.3
pip install scikit-learn==1.3.1
pip install -r requirements.txt
```

## Execution

Before training the models it is necessary to execute some scripts to generate the training files as shown below:

```
cd src/
python generate_dataset.py
```

Please note that the script `generate_dataset.py` has several parameters which can be configured depending on what is desired. For a first step let's use the default arguments provided.

The same will be done for the the script `generate_dataset_eval.py`, thus:

```
 python generate_dataset_eval.py
```

After the execution of the above scripts, the files `Beauty_sequential,straightforward_sequential_train.json` and `Beauty_sequential,straightforward_sequential_validation_seen:0.json` will be created under the folder `data/Beauty/`

In order to generate the training and validation files it is necessary to provide the `dataset` parameter to the scripts. For the `Electronics` dataset for example it follows:

```
python generate_dataset.py --dataset Electronics
python generate_dataset_eval.py --dataset Electronics
```

Aftewards, it is necessary to execute the file `train.py` as shown below:

```
python train.py
```

# Original Documentation
## Introduction

This repo presents OpenP5, an open-source platform for LLM-based Recommendation development, finetuning, and evaluation.  
> Paper: OpenP5: Benchmarking Foundation Models for Recommendation <br>
> Paper link: [https://arxiv.org/pdf/2203.13366.pdf](https://arxiv.org/pdf/2306.11134.pdf)

A relevant repo regarding how to create item ID for recommendation foundation models is available here:
> Paper: How to Index Item IDs for Recommendation Foundation Models <br>
> Paper link: https://arxiv.org/pdf/2305.06569.pdf <br>
> GitHub link: [https://github.com/Wenyueh/LLMforRS_item_representation](https://github.com/Wenyueh/LLMforRS_item_representation)

## News

-**[2023.9.16]** OpenP5 now supports both T5 and LLaMA-2 backbone LLMs.

-**[2023.6.10]** OpenP5 now supports 10 datasets and 3 item ID indexing methods for both sequential recommendation and straightforward recommendation tasks.

## Environment

Environment requirements can be found in `./environment.txt`

## Data Statistics

The statistics of the selected ten datasets can be found below:

| Datasets | ML-1M | Yelp| LastFM | Beauty | ML-100K |
|:-:|:-:|:-:|:-:|:-:|:-:|
| \#Users | 6,040 | 277,631 | 1,090 | 22,363 | 943 |
| \#Items | 3,416 | 112,394 | 3,646 | 12,101 | 1,349 |
|\#Interactions| 999,611 | 4,250,483 | 52,551 | 198,502 | 99,287 |
| Sparsity | 95.16\% | 99.99\% | 98.68\% | 99.93\% | 92.20\% |
| **Datasets** | **Clothing** | **CDs** | **Movies** | **Taobao** | **Electronics**|
| \#Users | 39,387 | 75,258 | 123,960 | 6,104 | 192,403 | 
| \#Items | 23,033 | 64,443 | 50,052 | 4,192 | 63,001 |
|\#Interactions| 278,677 | 1,697,533 | 1,697,533 | 46,337 | 1,689,188 |
|Sparsity| 99.97\% | 99.96\% | 99.97\% | 99.82\% | 99.99\% |

## More Results

More results on various datasets can be found in `./OpenP5_more_results.pdf`

## Usage

Download the data from [Google Drive link](https://drive.google.com/drive/folders/1W5i5ryetj_gkcOpG1aZfL5Y8Yk6RxwYE?usp=sharing), and put them into `./data` folder.

The training command can be found in `./command` folder. Run the command such as 

```
cd command
sh ML1M_random.sh
```

## Checkpoint

Download the checkpoint from [Google Drive Link](https://drive.google.com/drive/folders/19v7vgNBkIRdBm4FwPgHHiRz6Dnom29aR?usp=sharing), and put them into `./checkpoint` folder.

The evaluation command can be found in `./test_command folder`. Run the command such as 

```
cd ./test_command
sh ML1M_random.sh
```


## Citation

Please cite the following papers corresponding to the repository:
```
@article{xu2023openp5,
  title={OpenP5: Benchmarking Foundation Models for Recommendation},
  author={Shuyuan Xu and Wenyue Hua and Yongfeng Zhang},
  journal={arXiv:2306.11134},
  year={2023}
}
@article{hua2023index,
  title={How to Index Item IDs for Recommendation Foundation Models},
  author={Hua, Wenyue and Xu, Shuyuan and Ge, Yingqiang and Zhang, Yongfeng},
  journal={arXiv:2305.06569},
  year={2023}
}
```
