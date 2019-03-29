echo "Downloading BERT pre-train embedding model..."
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip
echo "Extracting BERT pre-train embedding model..."
unzip uncased_L-24_H-1024_A-16.zip

CORPUS="abstracts.txt"
BERT_BASE_DIR="./uncased_L-24_H-1024_A-16"
TF_RECORD="tf_examples.tfrecord"
TRAIN_OUTPUT="pretraining_output"

echo "Cloning google-research/bert"
git clone https://github.com/google-research/bert.git

echo "Creating pretraining data..."
python3 bert/create_pretraining_data.py \
  --input_file=$CORPUS \
  --output_file=$TF_RECORD \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5

echo "Running pretraining data..."
python3 bert/run_pretraining.py \
  --input_file=$TF_RECORD \
  --output_dir=$TRAIN_OUTPUT \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=20 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5

echo "Cleaning up..."
rm -rf uncased_L-24_H-1024_A-16.zip
