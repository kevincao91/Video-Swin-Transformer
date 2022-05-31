#!/usr/bin/env bash

cd ../../../

PYTHONPATH=. python tools/data/build_file_list.py tld3 data/tld3/videos_540_avi/ --level 2 --format videos --shuffle
echo "Filelist for videos generated."

cd tools/data/tld3/
