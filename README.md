This pipeline consists of 9 steps that are explained below:

**STEP 1**  : Create a funtion that does camera calibration
Camera lenses are prone to inherent distortions that can affect its perception of the real world. Taking account of this problem, as a starting step, camera calibration function will be applied so the car will get an accurate observation of the environment to navigate safely. Here, OpenCV's `cv2.findChessboardCorners` and `v2.drawChessboardCorners` are of great help to make this work.

**STEP 2**  : Distortion Correction

**STEP 3**  : Color Transform

**STEP 4**  : Perspective Transform

**STEP 5**  : Find lane boundary

**STEP 6**  : Calculate lane curvature

**STEP 7**  : Unwarp and draw entire lane boundaries

**STEP 8**  : Insert curvature and vehicle position value onto entire lane boundary image

**STEP 9**  : Test pipeline with a video
