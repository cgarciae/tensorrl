VIDEOS_PATH="$1"

for VIDEO in $(ls $VIDEOS_PATH); do
    VIDEO_PATH="$VIDEOS_PATH/$VIDEO"
    
    docker run \
        -v $(pwd)/$VIDEOS_PATH:/$VIDEOS_PATH \
        --user 1000 \
        jrottenberg/ffmpeg \
        -framerate 64 \
        -pattern_type glob \
        -i "/$VIDEO_PATH/*.jpg" \
        "/$VIDEO_PATH.mp4"
done
