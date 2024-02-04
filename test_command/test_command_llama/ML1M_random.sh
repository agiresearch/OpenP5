index="random"
dataset="ML1M"
backbone="open_llama_3b_v2"
bs=64
sample=0.2
path='../../data/'
epoch=2
lr=1e-3
wd=0.01
valid=0
checkpoint='../../model/'+${dataset}/${index}/${backbone}

CUDA_VISIBLE_DEVICES=0 python ../../src/src_llama/generate_llama.py --dataset ${dataset} --tasks sequential,straightforward --item_indexing ${index} --backbone ${backbone} --data_path ${path} --checkpoint_path ${checkpoint} --eval_batch_size 6 --lora 1 > ../log/${dataset}/eval_${dataset}_${backbone}_${index}_valid${valid}_lr${lr}_wd${wd}_${epoch}epoch_${bs}bs_sample${sample}.log