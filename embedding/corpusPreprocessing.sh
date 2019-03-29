echo "Downloading ACM-Citation-network V9..."
wget https://lfs.aminer.org/lab-datasets/citation/acm.v9.zip

echo "DBLP-Citation-network V10..."
wget https://lfs.aminer.cn/lab-datasets/citation/dblp.v10.zip

echo "Extracting ACM..."
unzip acm.v9.zip
echo "Extracting DBLP..."
unzip dblp.v10.zip

echo "Processing ACM..."
grep -p "^#\!" acm.txt | sed -e '/^#\!First Page of the Article/d' | sed 's/^#\!//' >>abstracts.txt

echo "Downloading jq..."
wget https://github.com/stedolan/jq/releases/download/jq-1.6/jq-linux64
mv jq-linux64 jq
chmod +x jq

echo "Processing DBLP..."
for file in dblp-ref/*; do
  time ./jq '.abstract' $file | sed -e '/null/d' | sed 's/\"//g' >>abstracts.txt
done

echo "Clean up..."
rm -rf acm.v9.zip acm.txt dblp.v10.zip dblp-ref
