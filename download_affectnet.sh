URL1=$1
URL2=$2

# Download affectnet and labels
mkdir -p ./data/affectnet
ZIP_FILE1=./data/affectnet.zip
wget -N $URL1 -O $ZIP_FILE1
unzip $ZIP_FILE1 -d ./data/affectnet
rm $ZIP_FILE1

ZIP_FILE2=./data/affectnet_labels.zip
wget -N $URL2 -O $ZIP_FILE2
unzip $ZIP_FILE2 -d ./data/affectnet
rm $ZIP_FILE2
