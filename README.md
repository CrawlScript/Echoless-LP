# Echoless-LP

Source code and dataset of the paper "[Echoless Label-Based Pre-computation for Memory-Efficient Heterogeneous Graph Learning](https://arxiv.org/abs/2511.11081)", which is accepted by AAAI 2026.



## Homepage and Paper

+ Homepage (Echoless-LP): https://github.com/CrawlScript/Echoless-LP
+ Paper Access:
    - **ArXiv**: [https://arxiv.org/abs/2511.11081](https://arxiv.org/abs/2511.11081) 


## Dataset Download


For DBLP, IMDB, and Freebase (from the HGB benchmark), please refer to the official repository for download instructions:  
- HGB benchmark: https://github.com/THUDM/HGB  
- Unzip the downloaded files into the `datasets` directory.

For OGBN-MAG, the code will automatically download the dataset via the `ogb` package.

For OAG-Venue and OAG-L1-Field, we follow the dataset preparation instructions from the NARS baseline, with minor file renaming:  
- Instructions: https://github.com/facebookresearch/NARS/tree/main/oag_dataset  
- After generating the `*.pk` and `*.npy` files:
  - Place these files in the `./datasets/nars_academic_oag/` directory.
  - Rename `graph_field.pk` to `graph_L1.pk`.




## Requirements

+ Linux
+ Python 3.7
+ torch==1.12.1+cu113
+ torchmetrics==0.11.4
+ dgl==1.0.2+cu113
+ ogb==1.3.5
+ shortuuid==1.0.11
+ pandas==1.3.5
+ gensim==4.2.0
+ numpy==1.21.6
+ tqdm==4.64.1
+ wandb==0.18.3



## Run Echoless-LP

You can run Echoless-LP with the following command:
```shell
sh scripts/run_DBLP.sh

sh scripts/run_Freebase.sh

sh scripts/run_IMDB.sh

sh scripts/run_OGBN-MAG.sh

sh scripts/run_OAG-Venue.sh

sh scripts/run_OAG-L1-Field.sh
```



## Cite

If you use Echoless-LP in a scientific publication, we would appreciate citations to the following paper:

```
@misc{hu2025echolesslabelbasedprecomputationmemoryefficient,
      title={Echoless Label-Based Pre-computation for Memory-Efficient Heterogeneous Graph Learning}, 
      author={Jun Hu and Shangheng Chen and Yufei He and Yuan Li and Bryan Hooi and Bingsheng He},
      year={2025},
      eprint={2511.11081},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.11081}, 
}
```
