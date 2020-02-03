import tensorflow as tf 
import cv2
import numpy as np

import posenet

N = 60

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

with tf.Session() as sess:
    model_cfg, model_outputs = posenet.load_model(args.model, sess)
    output_stride = model_cfg['output_stride']

lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (
                      cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
color = np.random.randint(0, 255, (100,3))


iteration = 0
while True:
    frame, display_frame, output_scale = posenet.read_cap(
        cap, scale_factor=0.7125, output_stride=)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Recalibrate to new skeleton every N frames
    if (iteration % N ==0) or (len(keypoint_coords)<2):
        heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
            model_outputs,
            feed_dict={'image:0': frame}
        )
        pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
            heatmaps_result.squeeze(axis=0),
            offsets_result.squeeze(axis=0),
            displacement_fwd_result.squeeze(axis=0),
            displacement_bwd_result.squeeze(axis=0),
            output_stride=output_stride,
            max_pose_detections=10,
            min_pose_score=0.15)

        keypoint_coords *= output_scale

        old_gray = frame_gray.copy()
        p0 = keypoint_coords
        mask = np.zeros_like(frame)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)
    cv2.imshow('frame',img)
    k = cv2.waitKey(1) & 0xff
    if k == ord('q'):
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    # Recalibrate to new skeleton every 15 frames
    p0 = good_new.reshape(-1,1,2)
    iteration += 1