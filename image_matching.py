#credits - https://github.com/dltpdn (Lee Sewoo)

import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = None
win_name = 'Camera Matching'
MIN_MATCH = 10
# ORB Detector generation  ---①
detector = cv2.ORB_create(1000)
# Flann Create extractor ---②
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6,
                   key_size = 12,
                   multi_probe_level = 1)
search_params=dict(checks=32)
matcher = cv2.FlannBasedMatcher(index_params, search_params)
# Camera capture connection and frame size reduction ---③
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if img1 is None:  # No image registered, camera bypass
        res = frame
    else:             # If there is a registered image, start matching
        img2 = frame
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        # Extract keypoints and descriptors
        kp1, desc1 = detector.detectAndCompute(gray1, None)
        kp2, desc2 = detector.detectAndCompute(gray2, None)
        # k=2로 knnMatch
        matches = matcher.knnMatch(desc1, desc2, 2)
        # Good Match Point Extraction with 75% of Neighborhood Distance---②
        ratio = 0.75
        good_matches = [m[0] for m in matches \
                            if len(m) == 2 and m[0].distance < m[1].distance * ratio]
        print('good matches:%d/%d' %(len(good_matches),len(matches)))
        # Fill the mask with zeros to prevent drawing all matching points
        matchesMask = np.zeros(len(good_matches)).tolist()
        # if More than the minimum number of good matching points
        if len(good_matches) > MIN_MATCH:
            # Find coordinates of source and target images with good matching points ---③
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ])
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ])
            # Find Perspective Transformation Matrix ---⑤
            mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            accuracy=float(mask.sum()) / mask.size
            print("accuracy: %d/%d(%.2f%%)"% (mask.sum(), mask.size, accuracy))
            if mask.sum() > MIN_MATCH:  # Set the mask to draw only outlier matching points if the normal
                # number is more than the minimum number of matching points
                matchesMask = mask.ravel().tolist()
                # Area display after perspective conversion to original image coordinates  ---⑦
                h,w, = img1.shape[:2]
                pts = np.float32([ [[0,0]],[[0,h-1]],[[w-1,h-1]],[[w-1,0]] ])
                dst = cv2.perspectiveTransform(pts,mtrx)
                img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        # Draw match points with mask ---⑨
        res = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, \
                            matchesMask=matchesMask,
                            flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    # Result output
    cv2.imshow(win_name, res)
    key = cv2.waitKey(1)
    if key == 27:    # Esc, 종료
            break
    elif key == ord(' '): # Set img1 by setting ROI to space bar
        x,y,w,h = cv2.selectROI(win_name, frame, False)
        if w and h:
            img1 = frame[y:y+h, x:x+w]
else:
    print("can't open camera.")
cap.release()
cv2.destroyAllWindows()
