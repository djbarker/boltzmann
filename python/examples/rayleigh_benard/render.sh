#!/usr/bin/bash

set -ex

export FPS=60
export DIR=./out_2

# First render the fields separately .
ffmpeg -framerate $FPS -i $DIR/temp_%06d.png -c:v libx264 -tune animation -crf 10 -r $FPS -y $DIR/temp.mp4
ffmpeg -framerate $FPS -i $DIR/vmag_%06d.png -c:v libx264 -tune animation -crf 10 -r $FPS -y $DIR/vmag.mp4

## Now add a pause at the end of each video.
PAUSE_SECS=3
ffmpeg -i $DIR/temp.mp4 -vf tpad=stop_mode=clone:stop_duration=$PAUSE_SECS -c:v libx264 -tune animation -crf 10 -y $DIR/temp_.mp4
ffmpeg -i $DIR/vmag.mp4 -vf tpad=stop_mode=clone:stop_duration=$PAUSE_SECS -c:v libx264 -tune animation -crf 10 -y $DIR/vmag_.mp4

## Lastly concat the videos together.
echo "#vids" > $DIR/vids.txt
echo "file 'temp_.mp4'" >> $DIR/vids.txt
echo "file 'vmag_.mp4'" >> $DIR/vids.txt
ffmpeg -f concat -safe 0 -i $DIR/vids.txt -c copy -y rb3.mp4

## Finally clean up.
# rm $DIR/temp.mp4 $DIR/vmag.mp4 $DIR/temp_.mp4 $DIR/vmag_.mp4 $DIR/vids.txt