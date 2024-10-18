#!/bin/bash

# Clone the latest NNCF
WORKING_DIR=/tmp/hf_compression_service
rm -rf $WORKING_DIR
git clone https://github.com/openvinotoolkit/nncf.git $WORKING_DIR

# Install the environment
pip install -U pip
cd $WORKING_DIR/nncf/tools/
pip install -r requirements.txt
pip install ../../

cd $WORKING_DIR
# Don't forger to put encryption passwork to /etc/nncf-service/secret.key
sh ./tools/hf_compression_service/start_service.sh 2>&1 > service.log

