# OpenP5: Benchmarking Foundation Models for Recommendation

## Introduction

This repo presents OpenP5, an open-source library for benchmarking foundation models for recommendation under the Pre-train, Personalized Prompt and Predict Paradigm (P5).  
> Paper link: [https://arxiv.org/pdf/2203.13366.pdf](https://arxiv.org/pdf/2306.11134.pdf)

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

Please cite the following paper corresponding to the repository:
```
@article{xu2023openp5,
  title={OpenP5: Benchmarking Foundation Models for Recommendation},
  author={Shuyuan Xu and Wenyue Hua and Yongfeng Zhang},
  journal={arXiv preprint arXiv:2306.11134},
  year={2023}
}
```
