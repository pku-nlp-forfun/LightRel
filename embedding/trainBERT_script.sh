for file in `ls dblp-ref`
do time ../jq-linux64 '.abstract' dblp-ref/$file | sed -e '/null/d' | sed 's/\"//g' >> abstracts.txt
done
echo "cd ~ && bash tar.sh" | qsub -N tar1 -d ~/output -l walltime=24:00:00 -l nodes=5:ppn=2
echo "cd ~/bert && bash create_pretraining_data.sh" | qsub -N embedding_bert6 -d ~/output -l walltime=24:00:00 -l nodes=5:ppn=2
echo "cd ~/fastText && make" | qsub -N fasttext_build1 -d ~/output -l walltime=24:00:00 -l nodes=5:ppn=2
echo "cd ~/fastText && make" | qsub -N fasttext_build1 -d ~/output -l walltime=24:00:00 -l nodes=5:ppn=2
time ./fasttext skipgram -input sementic/dblp_abstracts.txt -output model_v10 -dim 300 -minCount 5 -thread 30
echo "cd ~ && time ./fasttext skipgram -input sementic/dblp-semeval.txt -output model_acm -dim 300 -minCount 5 -thread 30" | qsub -N fasttext3 -d ~/output -l walltime=24:00:00 -l nodes=5:ppn=2

echo "cd ~ && tar -czvf bert_acm.tar.gz ~/uncased_L-24_H-1024_A-16/acm_dblp" | qsub -N bert_acm1 -d ~/output
echo "cd ~/bert && bash pretrain.sh" | qsub -N pretraining2 -d ~/output -l walltime=24:00:00 -l nodes=5:ppn=2

BERT_BASE_DIR=../uncased_L-24_H-1024_A-16
python3 create_pretraining_data.py \
  --input_file=../sementic/abstracts.txt \
  --output_file=tf_examples.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5

BERT_BASE_DIR="../uncased_L-24_H-1024_A-16"
python run_pretraining.py \
  --input_file=./tf_examples.tfrecord \
  --output_dir=$BERT_BASE_DIR \
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
for file in `ls dblp-ref`
do time ../jq-linux64 '.abstract' dblp-ref/$file | sed -e '/null/d' | sed 's/\"//g' >> dblp_abstracts.txt
done
