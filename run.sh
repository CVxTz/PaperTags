
#mkdir /media/jenazzad/Data/ML/paper_tags/data
#mkdir /media/jenazzad/Data/ML/paper_tags/split
#mkdir /media/jenazzad/Data/ML/paper_tags/predictions

# python paper_tags/steps/download_json.py --target_path /media/jenazzad/Data/ML/paper_tags/data

#python paper_tags/steps/split_data.py --target_path /media/jenazzad/Data/ML/paper_tags/split \
#                                      --data_path /media/jenazzad/Data/ML/paper_tags/data/papers-with-abstracts.json

#python paper_tags/steps/train_classifier.py --train_path /media/jenazzad/Data/ML/paper_tags/split/train.json \
#                                            --val_path /media/jenazzad/Data/ML/paper_tags/split/val.json \
#                                            --mlb_path /media/jenazzad/Data/ML/paper_tags/split/mlb.joblib \
#                                            --model_path /media/jenazzad/Data/ML/paper_tags/model

python paper_tags/steps/val_inference.py  --val_path /media/jenazzad/Data/ML/paper_tags/split/val.json \
                                          --model_path /media/jenazzad/Data/ML/paper_tags/model \
                                          --output_path /media/jenazzad/Data/ML/paper_tags/predictions

