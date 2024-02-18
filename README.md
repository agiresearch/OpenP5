# OpenP5: An Open-Source Platform for Developing, Training, and Evaluating LLM-based Recommender Systems

## Introduction

This repo presents OpenP5, an open-source platform for LLM-based Recommendation development, finetuning, and evaluation.  
> Paper: OpenP5: Benchmarking Foundation Models for Recommendation <br>
> Paper link: [https://arxiv.org/pdf/2203.13366.pdf](https://arxiv.org/pdf/2306.11134.pdf)

A relevant repo regarding how to create item ID for recommendation foundation models is available here:
> Paper: How to Index Item IDs for Recommendation Foundation Models <br>
> Paper link: https://arxiv.org/pdf/2305.06569.pdf <br>
> GitHub link: [https://github.com/Wenyueh/LLM-RecSys-ID](https://github.com/Wenyueh/LLM-RecSys-ID)

## News
-**[2023.12.20]** We have made the first release of the project under the release-1.0 branch, which is also provided as the release-1.0 under the Release section of the project. This is a complete and readily executable branch that can help you to quickly get things running and do experiments for both T5 and LLaMA backbones. However, these two backbones are implemented as two separate python files. Currently, we are further refactoring the code to make T5 and LLaMA backbones compatible in the same codebase structure, and we will make the second release once that is finished.

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


## Usage

Download the data from [Google Drive link](https://drive.google.com/drive/folders/1W5i5ryetj_gkcOpG1aZfL5Y8Yk6RxwYE?usp=sharing), and put them into `./data` folder.

Run the following command to generate all data

```
sh generate_dataset.sh
```

The training command can be found in `./command` folder. Run the command such as 

```
cd command
sh ML1M_t5_sequential.sh
```

## Checkpoint

The evaluation command can be found in `./test_command folder`. Run the command such as 

```
cd ./test_command
sh ML1M_t5_sequential.sh
```


## Citation

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
  journal={SIGIR-AP},
  year={2023}
}
```
