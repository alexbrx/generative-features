URL=$1

# Download affectnet
ZIP_FILE=./data/affectnet.zip
mkdir -p ./data/
wget -N $URL -O $ZIP_FILE
unzip $ZIP_FILE -d ./data/
rm $ZIP_FILE

fi
