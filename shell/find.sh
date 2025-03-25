export CUDA_VISIBLE_DEVICES=0

python -m test.find_problem\
    --test_file /E:/yanzuWu/train_split.csv\
    --knowledge_base /E:/infoseek_100k_wiki/wiki_100_dict_v4.json\
    --model_name clip\
    --top_k 10