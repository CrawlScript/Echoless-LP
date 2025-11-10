# Source code for Echoless-LP

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




