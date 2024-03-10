python3 tracking.py \
-f exp.py \
-c pretrained/bytetrack_x_mot20.tar \
--path videos/easy_9.mp4 \
--nms 0.7 \
--track_thresh 0.7 \
--track_buffer 90 \
--match_thresh 0.8 \
--mot20 \
--fuse