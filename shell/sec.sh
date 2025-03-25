export CUDA_VISIBLE_DEVICES=0

python -m test.second_round\
    --test_file /E:/yanzuWu/train_split.csv\
    --knowledge_base /E:/encyclopedic_kb_wiki_parsed.json\
    --result_file /E:/yanzuWu/retrieval_result.json\
    --model_name sbert\
    --top_ks 1,5,10,20\

