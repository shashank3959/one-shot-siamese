#!/bin/sh

# unzip images
unzip '*.zip'

# move zip files to raw dir
mkdir raw/
mkdir processed/
mv '*.zip' raw/

# rename folders
mv images_background/ background/
mv images_evaluation/ evaluation/

# move 10 first evaluation subdirs to background dir
pushd evaluation/
folders=(*/)
popd
for ((i=0; i<10; i++))
do
  mv "evaluation/${folders[i]}" background/
done

echo "done"
