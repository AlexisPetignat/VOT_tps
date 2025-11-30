# Visual Object Tracking: Project report on TPs 1 - 4
### Author: Petignat Alexis, 2025

This report will present the work done during the four practical sessions (TPs) of the Visual Object Tracking course. The main objective of these TPs was to implement and evaluate various object tracking algorithms on video sequences, such as Kalman Filter, IoU based tracking and Deep Learning based tracking. Below is a summary of each TP, the implementation process and the difficulties encountered.

## TP1: Kalman Filter for Object Tracking

Kalman Filter is a recursive algorithm used for estimating the state of a dynamic system, based on a series of noisy measurements. The goal of this TP was to implement a basic Kalman Filter for tracking a single object in an empty video scene.

### Implementation
- The first task of this TP was to build the KalmanFilter structure, which included defining the state vector, measurement vector, and the necessary matrices (state transition, observation, process noise covariance, and measurement noise covariance). The state vector was defined to include the position and velocity of the object. No particular difficulty arose in this part.

- The next task was to implement the prediction function, which, due to the simple implementation of the filter, was straightforward. The state was predicted using the state transition matrix. The prediction was passed through the state matrix, which is not necessary for the update function.

- The update function was the next task. This is where issues started arising. I had some issues with the theory behind the function, which caused several wrong implementations of the function. The main issue was how the coordinates were passed to the function, which was a squeezed array, causing a dimension mismatch that took me some time to debug. Once this was fixed, the function started working correctly. 

- Finally, the Kalman Filter was integrated into the tracking pipeline. The filter was initialized with the first detection of the object, and then for each subsequent frame, the prediction and update functions were called to track the object. The results were visualized by drawing bounding boxes around the tracked object. At first, the predicted box was moving on a straight line at the bottom of the screen, which was due to a wrong indexing in the update method which was quickly fixed. Once this was fixed, the tracker started working as intended but I noticed somethiing was off with the prediction, which always seemed to lag behind the actual position of the object. After some debugging, I realized I was passing my predicted state from the step before instead of my updated state from this state. Once this was fixed, the tracker worked perfectly as expected.

### Results
The Kalman Filter tracker was able to successfully track the object throughout the video sequence. The predicted bounding boxes closely followed the actual position of the object, demonstrating the effectiveness of the Kalman Filter for object tracking in a simple scenario. Below is a visualization of the tracking results. The red box represents the actual detected position of the object, while the blue box represents the predicted position from the Kalman Filter.
![Kalman Filter Tracking Results](./2D_Kalman-Filter_TP1/video.avi)


## TP2: Multi-Object Tracking using IoU-based Data Association

IoU (Intersection over Union) is a metric used to evaluate the overlap between two bounding boxes. In this TP, the goal was to implement a multi-object tracking algorithm using IoU-based data association, coupled with the Hungarian algorithm for data association. The detections in each scene were provided by a given csv file which was loaded and parsed to extract the bounding boxes for each frame.

### Implementation
- The first step was to understand where to actually start with this problem. The first idea that came to mind was creating several structures to ease the implementation. I created 3 classes:
1. `Point`: This class represents a single point in a 2D space, including its x and y coordinates stored as integers.
2. `BoundingBox`: This class represents a bounding box defined by its top-left and bottom-right corners stored as Points defined above.
3. `Track`: This class represents a single tracked object, including an ID for the track, a bounding box, and a counter for the number of consecutive frames where the object was not detected (missed frames), used to determine when the track is considered as dead.
4. `TrackList`: This class was originally created to manage a list of tracks, including adding new tracks, updating existing tracks, and removing dead tracks. However, I ended up not using it as I found it easier to manage the list of tracks directly with a python list, which ended up working smoothly.

- Once these structures were created, I started implementing the main tracking loop. First, I loaded all the detections and mapped them to bounding boxes. I also implemented a function that, for two given bounding boxes, computes their IoU. This function was straightforward to implement and did not pose any issues thanks to stack overflow.

- Next, I handled the starting case where no tracks were present. In this case, I simply created a new track for each detection in the current frame.
- For subsequent frames, I computed the IoU between each existing track and each detection, storing the results in an IoU matrix. The cost matrix was obtained with $1 - \text{IoU}$.
- I then used the Hungarian algorithm to find the optimal assignment between tracks and detections based on the cost matrix. This was implemented using the `scipy.optimize.linear_sum_assignment` function, which made it easy to implement without any issues. I did get a problem with me getting confused with the IDs from the hungarian algorithm, the track IDs, and the detection IDs since I did not know which one to associate, causing tracks to constatntly change their IDs. After some debugging, I realized I had to use the indices returned by the Hungarian algorithm to map back to the original track and detection IDs.
- The result was correct. For better associations, I set a threshold on the IoU values to determine whether a track and detection should be associated. If the IoU was below this threshold (set to 0.3), the association was discarded and no box was displayed.
- The final result works very well. The tracks are well associated with the detections and the IDs on the video are coherent. The only issue I noticed is that when two objects cross paths, the tracker sometimes swaps their IDs. The boxes are also shaking a bit, which is normal since no prediction is done between frames. Overall, the tracker performs well in this multi-object tracking scenario with given detections.
- Finally, each detection from the file was associated with a track and stored in a separate file, stored in ./ADL-Rundle-6/gt/tp2_det.txt.

### Results
The IoU-based multi-object tracker was able to successfully track multiple objects throughout the video sequence. The bounding boxes were correctly associated with the detected objects, and the tracker was able to maintain consistent IDs for each object. Below is a visualization of the tracking results. Each bounding box is labeled with its corresponding track ID.
![IoU-based Multi-Object Tracking Results](./ADL-Rundle-6/tp2/video.avi)


## TP3: Kalman Filter with IoU-based Data Association for Multi-Object Tracking

This TP is the combination of the first two TPs. The goal was to upgrade the IoU-based multi-object tracker from TP2 by integrating a Kalman Filter for each track to improve the tracking performance. The Kalman Filter would provide predictions for the object positions, which would then be used in conjunction with the IoU metric for data association.

### Implementation
- The first step was to re-organize my code. It was messy already, and not quite fit to integrate more logic. I extracted the cost matrix computation in a separate function to remove logic from the main loop, as well as the frame updating logic (adding rectangles for the bounding boxes).
- Next, I changed the BoundingBox class to include a method to compute the center of the box, which would be used as the measurement for the Kalman Filter. I also added a method to shift the box to a new center, which would be used to update the bounding box based on the Kalman Filter's prediction. The Track class also gained a new attribute, which is the Kalman Filter instance associated with the track.
- Then began the hardest part. The implemention of this system was quite difficult since it required to separate detections into 3 categories: matched detections, unmatched detections, and unmatched tracks that each required a specific handling.
1. `Matched Detections`: Matched detections were the first case I handled. These detections can be defined as those with a great IoU with an existing track. For these detections, I updated the corresponding Kalman Filter with the detection's center as the measurement. After updating the Kalman Filter, I obtained the predicted state and used it to update the bounding box of the track. The updated box was displayed on the frame with its ID, and the predicted box was stored as the track's bounding box for better IoU score at the next step. I did not encounter issues with this part.
2. `Unmatched Detections`: Unmatched detections were the next case I handled, and also the trickiest one. These detections can be defined as those with a low IoU with all existing tracks, and can be considered as new objects entering the scene. For these detections, a new track is created and initialized with a new Kalman Filter with the detection's center as the initial state. The new track was added to the list of existing tracks, and its bounding box was displayed on the frame with a new ID. The main issue I encountered here was detecting which detections were unmatched. I initially tried to do this by checking the IoU values, but it was error-prone and led to incorrect associations quite often. The solution I opted for was to store all detections that were matched during the previous step, and then simply check which detections were not in this list. This approach worked much better, once I figured out a way to manage the track IDs in a better way, since my previous implementation did not allow to assign IDs to tracks the way I needed.
3. `Unmatched Tracks`: Unmatched tracks were the final case I handled. This was also the easiest one by far. The intended way to handle unmatched tracks is to increase their "missed frames" counter, and if this counter exceeds a certain threshold, the track is considered dead and removed from the list of existing tracks. But instead of doing this, I simply considered all tracks as unmatched at the beginning of the step, increasing all the missed counters by one. Then, during the matched detections step, I reset the counter for matched tracks back to zero. Finally, at the end of the loop, I removed all tracks with a missed frames counter exceeding a threshold and for thoses which did not reach this threshold, I displayed the prediction of the Kalman Filter on the frame, to infer the object's position. This approach worked well and simplified the logic significantly.

- Overall, the integration of the Kalman Filter into the IoU-based multi-object tracker improved the tracking performance, resulting in smoother trajectories and more consistent IDs for each object. Although, I noticed that some boxes would appear out of nowhere and go across the screen with high velocity. This was most likely due to detections of a weird object for a few frames, causing the Kalman Filter to predict a high velocity for the object (whether it actually existed or not). To fix this, I added a counter to the Track class that counts the total number of frames where a detection was matched with the track. This allowed to set up a threshold to assert that all predictions could only be displayed if the track had been matched at least a certain number of times, which guaranteed that the track was actually valid, and not some "phantom" track.

### Results
The Kalman Filter with IoU-based data association multi-object tracker was able to successfully track multiple objects throughout the video sequence. The integration of the Kalman Filter improved the tracking performance, resulting in smoother trajectories and more consistent IDs for each object. Below is a visualization of the tracking results. Each bounding box is labeled with its corresponding track ID. When a bounding box is inferred, the word "Predicted" is displayed next to the ID.
The result is visibly better than the previous TP, with less shaking boxes and better ID consistency. However, some phantom boxes can still appear from time to time when many objects are on the scene or when objects cross paths.
![Kalman Filter with IoU-based Multi-Object Tracking Results](./ADL-Rundle-6/tp3/video.avi)


## TP4: Deep Learning-based Object Tracking with IoU Data Association and Kalman Filter

This TP was to go even further beyond the previous TPs by integrating a Deep Learning-based object detector into the tracking pipeline. The goal was to use a pre-trained object detection model to provide a similarity score for data association, instead of relying solely on the IoU metric. The Kalman Filter would still be used for predicting object positions between frames.


### Implementation

The implementation of this TP was quite straightforward since it was mostly reusing the code from TP3 with some modifications to integrate the Deep Learning-based object detector.
- The first step was to understand how to use and call the model. This model was a provided onnx model, which we already had experience with in previous courses. The goal of this model was not to provide a similarity score between two bounding boxes, but rather to provide a feature vector for each bounding box. The similarity score could then be computed as a similarity function between two feature vectors, here defined as the cosine similarity score.
- The changes in the code were minimal. The only function I changed is the function that computes the cost matrix, which now also computes the cosine similarity for each pair of track and detection feature vectors. The cost matrix was then computed as a weighted sum of the IoU cost and the cosine similarity cost, with weights defined as hyperparameters ALPHA and BETA (set to 0.7 and 0.3, respectively).
- The feature extraction process required some preprocessing beforehand. First, each bounding box was cropped from the frame and resized to the input size expected by the model (64x128). The cropped image was then normalized using the mean and standard deviation values provided in the TP instructions. Finally, the preprocessed image was passed through the model to obtain the feature vector. I had several issues with this part, due to image boxes being out of bounds of the image, especially with predictions since these are inferred and fixed on the velocity of the object. Several fixed were required to prevent this. I clamped the box coordinates to always be positive and stay in bounds of the image. This was not a trivial task since I had implemented operators for Points, which also had to be changed. Another issue I faced were boxes with zero area (corners sharing an x or y coordinate), which caused errors during resizing. To fix this, I added a check to ensure that the width and height of the box were at least 1 pixel before cropping and resizing (this would only occur on the edges of the picture, and most likely apply to only objects which left the scene).
- At this point, the code was working but it was extremely slow. I ended up realizing that calling a forward pass on the model for each couple of box, for each frame, was extremely inefficient. To fix this, I decided to precompute all feature vectors for all detections in the current frame before computing the similarity matrix in one call to the cosine similarity function from sklearn. This change drastically improved the speed of the tracker.


### Results
The Deep Learning-based object tracker with IoU data association and Kalman Filter was able to successfully track multiple objects throughout the video sequence. The integration of the Deep Learning-based object detector improved the data association process, resulting in more accurate tracking and less ID switching between objects. Below is a visualization of the tracking results. Each bounding box is labeled with its corresponding track ID. When a bounding box is inferred, the word "Predicted" is displayed next to the ID.
![Deep Learning-based Multi-Object Tracking Results](./ADL-Rundle-6/tp4/video.avi)


# Conclusion

These TPs were a great opportunity to learn and implement various object tracking algorithms. Starting from a basic Kalman Filter for single-object tracking, we progressed to more complex multi-object tracking scenarios using IoU-based data association and finally integrated Deep Learning-based object detection for improved tracking performance. My personal favorite was TP3, since it already felt like a high quality object tracking system, with smooth predictions and good ID consistency. TP4 was also interesting, but the performance gain was not as significant as I expected, and the implementation was extremely straightforward compared to TP3. I liked a lot working on these TPs, they were challenging but rewarding, and I feel that I have gained a solid understanding of object tracking techniques and their practical applications.