#!/bin/bash
# -l h_rt=72:00:00


# for every tar.gz file in the current directory, untar it

cd /exp/amartin/scale24/info
for f in *.tar.gz
do
    echo "Untarring $f"
    tar -xzf $f
    rm internvid/info/*/*.jpg
    rm internvid/info/*/*.webp
done



