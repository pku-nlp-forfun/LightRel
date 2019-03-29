CORPUS="abstracts.txt"

echo "Setting up fastText..."
wget https://github.com/facebookresearch/fastText/archive/v0.2.0.zip
unzip v0.2.0.zip
cd fastText-0.2.0
make

echo "Training fastText..."
cd ..
fastText-0.2.0/fasttext skipgram -input $CORPUS -output model -dim 300 -minCount 5
