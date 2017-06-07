#!/usr/bin/env bash

# grab up-to-date Python 3
apt-get update
apt-get install python3-pip python3-dev
# install Tensorflow (CPU only)
pip3 install tensorflow

# grab pre-trained embeddings
mkdir -p /vagrant/data/embeddings
if [ ! -e /vagrant/data/embeddings/PubMed_CBOW.bin ]; then
    cd /vagrant/data/embeddings && wget http://slate.cse.ohio-state.edu/BMASS/PubMed_CBOW.bin
fi
if [ ! -e /vagrant/data/embeddings/PubMed_SGNS.bin ]; then
    cd /vagrant/data/embeddings && wget http://slate.cse.ohio-state.edu/BMASS/PubMed_SGNS.bin
fi
if [ ! -e /vagrant/data/embeddings/PubMed_Glove.bin ]; then
    cd /vagrant/data/embeddings && wget http://slate.cse.ohio-state.edu/BMASS/PubMed_Glove.bin
    wget http://slate.cse.ohio-state.edu/BMASS/PubMed_Glove_vocab
fi
# grab frequent term file
if [ ! -e /vagrant/data/PubMed_frequent_terms.txt ]; then
    cd /vagrant/data && wget http://slate.cse.ohio-state.edu/BMASS/PubMed_frequent_terms.txt
fi

# make results directories
for d in all_info multi_answer single_answer; do
    mkdir -p /vagrant/data/results/$d
done
mkdir -p /vagrant/data/logs

# copy the pre-built config and make files
cp /vagrant/config.py /vagrant/src/
cp /vagrant/makefile /vagrant/src/

BASHRC=/home/ubuntu/.bashrc
touch $BASHRC
echo "cat <<EOF" >> $BASHRC
cat /vagrant/welcome_msg >> $BASHRC
echo "EOF" >> $BASHRC
