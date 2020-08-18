This pipeline consists of 9 steps that are explained below:
1. **Create a funtion that does camera calibration**
Camera lenses are prone to inherent distortions that can affect its perception of the real world. Taking account of this problem, as a starting step, camera calibration function will be applied so the car will get an accurate observation of the environment to navigate safely. Here, OpenCV's `cv2.findChessboardCorners` and `v2.drawChessboardCorners` are of great help to make this work.

2. Distortion Correction

3. Color Transform

4. Perspective Transform

5. Find lane boundary

6. Calculate lane curvature

7. Unwarp and draw entire lane boundaries

8. Insert curvature and vehicle position value onto entire lane boundary image

9. Test pipeline with a video
