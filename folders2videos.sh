VIDEOS_PATH="$1"

for VIDEO in $(ls $VIDEOS_PATH); do
    VIDEO_PATH="$VIDEOS_PATH/$VIDEO"
    
    ffmpeg \
        -framerate 25 \
        -pattern_type glob \
        -i "$VIDEO_PATH/*.jpg" \
        "$VIDEO_PATH.mp4"
done
