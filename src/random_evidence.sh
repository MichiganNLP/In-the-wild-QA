
pred_num=5

python -m src.main random_evidence \
    --train_data example_data/wildQA-data/train.json \
    --dev_data example_data/wildQA-data/dev.json \
    --test_data example_data/wildQA-data/test.json \
    --pred_num ${pred_num}
