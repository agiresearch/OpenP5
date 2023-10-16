for dataset in Beauty ML100K ML1M Yelp Electronics Movies CDs Clothing Taobao LastFM
do
    for indexing in random sequential collaborative
    do
        python ./src/generate_dataset.py --dataset ${dataset} --data_path ./data/ --item_indexing ${indexing} --tasks sequential,straightforward --prompt_file ./prompt.txt

        python ./src/generate_dataset_eval.py --dataset ${dataset} --data_path ./data/ --item_indexing ${indexing} --tasks sequential,straightforward --prompt_file ./prompt.txt --mode validation --valid_prompt seen:0

        python ./src/generate_dataset_eval.py --dataset ${dataset} --data_path ./data/ --item_indexing ${indexing} --tasks sequential,straightforward --prompt_file ./prompt.txt --mode test --test_prompt seen:0

        python ./src/generate_dataset_eval.py --dataset ${dataset} --data_path ./data/ --item_indexing ${indexing} --tasks sequential,straightforward --prompt_file ./prompt.txt --mode test --test_prompt unseen:0
    done
done
