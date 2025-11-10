# Set default values
START_SEED=${1:-1} # Default start seed is 1
END_SEED=${2:-1}   # Default end seed is 1
GPU=${3:-0}     # Default GPU ID is 0



DATASET=imdb
METHOD=rphgnn
OUTPUT_DIR=results/final_results/${DATASET}/${METHOD}/

USE_RENORM=True


for SEED in $(seq $START_SEED $END_SEED)
do

for EMBEDDING_SIZE in 1024
do

for FEAT_MODE in all_feat
do

for LABEL_K in 8
do

for NUM_PARTITIONS in 4
do

for NUM_LP_REPEATS in 3
do

for LABEL_MERGE_MODE in append
do

for LABEL_INPUT_DROP_RATE in 0.8
do

for INPUT_DROP_RATE in 0.9 
do

for DROP_RATE in 0.7 
do

for USE_EXTRA_MASK in True 
do



USE_NRL=False 
TRAIN_STRATEGY=common
USE_INPUT=True
ALL_FEAT=True 

HIDDEN_SIZE=512



SQUASH_K=3
EPOCHS=500
MAX_PATIENCE=50
USE_LABEL=True
EVEN_ODD="all"

LABEL_EMB_SIZE=-1 

python -u main_echoless_lp.py \
    --method ${METHOD} \
    --dataset ${DATASET} \
    --use_nrl ${USE_NRL} \
    --use_label ${USE_LABEL} \
    --even_odd ${EVEN_ODD} \
    --train_strategy ${TRAIN_STRATEGY} \
    --use_input ${USE_INPUT} \
    --input_drop_rate ${INPUT_DROP_RATE} \
    --drop_rate ${DROP_RATE} \
    --label_input_drop_rate ${LABEL_INPUT_DROP_RATE} \
    --hidden_size ${HIDDEN_SIZE} \
    --squash_k ${SQUASH_K} \
    --label_k ${LABEL_K} \
    --num_partitions ${NUM_PARTITIONS} \
    --use_extra_mask ${USE_EXTRA_MASK} \
    --use_renorm ${USE_RENORM} \
    --num_epochs ${EPOCHS} \
    --max_patience ${MAX_PATIENCE} \
    --embedding_size ${EMBEDDING_SIZE} \
    --label_merge_mode ${LABEL_MERGE_MODE} \
    --label_emb_size ${LABEL_EMB_SIZE} \
    --use_all_feat ${ALL_FEAT} \
    --feat_mode ${FEAT_MODE} \
    --num_lp_repeats ${NUM_LP_REPEATS} \
    --output_dir ${OUTPUT_DIR} \
    --seed ${SEED} \
    --gpus ${GPU}  


done
done
done
done
done
done
done
done
done
done
done