# python ../src/main.py --datasets Clothing --distributed 1 --gpu 6,7 --tasks sequential,straightforward --item_indexing sequential --epochs 20 --batch_size 64 --master_port 1124 --prompt_file ../prompt.txt --sample_prompt 1 --eval_batch_size 20 --dist_sampler 0 --max_his 20  --sample_num 3,3 --train 0 --test_prompt seen:0 --lr 1e-3 --test_filtered 0 --model_name Clothing_sequential.pt

python ../src/main.py --datasets Clothing --distributed 1 --gpu 6,7 --tasks sequential,straightforward --item_indexing sequential --epochs 20 --batch_size 64 --master_port 1124 --prompt_file ../prompt.txt --sample_prompt 1 --eval_batch_size 20 --dist_sampler 0 --max_his 20  --sample_num 3,3 --train 0 --test_prompt unseen:0 --lr 1e-3 --test_filtered 0 --model_name Clothing_sequential.pt

