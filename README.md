# Project overview

In this project, I wrote a software pipeline that identifies lane boundaries in a specific video taken from a front facing camera mounted on a vehicle. Frames from the video were taken and used to extract enough information while creating this pipeline. All details in this pipeline can be seen in [this depository](https://github.com/Arina-W/Detecting-Lane-Boundaries).

## Directories Structure:
---

- `camera_cal`    : Compilation of chessboards images for camera calibration purpose
- `output_images` : Compilation of output images from each step of pipeline
- `test_images`   : Compilation of test images taken from video to extract information
- `videos`        : Compilation of test video and output result of pipeline


## Creating a great writeup:
---
A great writeup should include the rubric points as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

## The Project
---

### This pipeline consists of 9 steps that are explained below:

### **1. Create a funtion that initiates camera calibration**
 * Camera lenses are prone to inherent distortions that can affect its perception of the real world. 
 * Taking account of this problem, as a starting step, camera calibration function will be applied so the car will get an accurate observation of the environment to navigate safely. 
 * Here, OpenCV's `cv2.findChessboardCorners()` and `cv2.calibrateCamera()` are used to get information for calibration purpose.
 ```
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
        
        objpoints = [] # for 3d points in real world 
        imgpoints = [] # for 2d points in image world
        
        if found:
            objpoints.append(objp)
            imgpoints.append(corners)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)        
 ```
 * Corners found during calibration attempt will return a few parameters, including `mtx` and `dist`, which will be useful for the following step.


### **2. Distortion Correction**
 * This step is crucial to ensure that the geometrical shape of objects are represented consistently, no matter where they appear in the image.
 * To achieve this, OpenCV function `cv2.undistort()` are applied to compute the calibration of camera and undistortion using Matrix, `mtx`, value and distortion coefficient, `dist`, obtained from step one. 
 * Below is the original image(before) and its undistorted image(after), after applying both step 1 and 2 of pipeline.
 
 ![step2](https://github.com/Arina-W/Detecting-Lane-Boundaries/blob/master/output_images/camera_calibration.png)
 
 * Below is the result of calibration done on a test image taken from the video.
 
 ![step2.1](https://github.com/Arina-W/Detecting-Lane-Boundaries/blob/master/output_images/camera_calibration_on_testimage.png)
 


### **3. Color Transform**
 * Gradients and color spaces taken off of an image offer great advantages to find pixels that form the lines in the video. 
 * In this step a gradient threshold using `cv2.Sobel` operator in x the direction are used since it does a cleaner job to pick up the lane lines in the image.
```
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    satchannel = hls[:,:,2] # saturation channel
    
    # Sobel of x-axis direction using saturation channel 
    sobelx = cv2.Sobel(satchannel, cv2.CV_64F, 1, 0, 3) # calculate derivative in x direction
    absox = np.absolute(sobelx) # magnitude derivative to stand out from horizontal (y-axis)
    scaledsobelx = np.uint8(255*absox/np.max(absox)) # convert value image to 8-bit
    
    # Threshold sobel channel
    sobelxbinary = np.zeros_like(scaledsobelx)
    sobelxbinary[(scaledsobelx >= gradientthres[0]) & (scaledsobelx <= gradientthres[1])] = 1
```
 * In combination of the Sobel operator, a threshold of S channel of HLS color space are also taken to filter out better look of the lines.
```
    # Threshold saturation channel 
    satbinary = np.zeros_like(satchannel)
    satbinary[(satchannel >= satthres[0]) & (satchannel <= satthres[1])] = 1
    
    # COMBINE
    combined = np.zeros_like(satchannel)
    combined[(sobelxbinary == 1) | (satbinary == 1)] = 1
```

 * Below is the output of a transformed image after step 1, 2, and 3.
 
 ![step3](https://github.com/Arina-W/Detecting-Lane-Boundaries/blob/master/output_images/Color_and_gradient_transformed.png)



### **4. Perspective Transform**
 * This step maps 4 points in a test image to a different(desired) angle with a new perspective. The transformation that is beneficial for this project is the *birds's-eye view* transform.
 * Here, OpenCV function `cv2.getPerspectiveTransform()` and `cv2.warpPerspective()` are used to transform the area of the desired 4 points to a top-down view.


### **5. Find lane boundary**
 * 

### **6. Calculate lane curvature**
 * 

### **7. Unwarp and draw entire lane boundaries**
 * 

### **8. Insert curvature and vehicle position value onto entire lane boundary image**
 * 

### **9. Test pipeline with a video**
 * 
