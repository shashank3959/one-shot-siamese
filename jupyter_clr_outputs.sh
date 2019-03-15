#!/bin/bash

filename=$@
echo "You provided" "$filename"

jupyter nbconvert --ClearOutputPreprocessor.enabled=True   --to notebook --output=$filename $filename

