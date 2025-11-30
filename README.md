# Visual Object Tracking: Project report on TPs 1–4
### Author: name, 2025

This report presents the work done during the four practical sessions (TPs) of the Visual Object Tracking course. The main objective was to implement various object tracking algorithms on video sequences, such as the Kalman Filter, IoU-based multi-object tracking, and deep learning–based tracking. Below is a summary of each TP, the implementation process and the difficulties encountered.

---

## TP1: Kalman Filter for Object Tracking

The Kalman Filter is a recursive algorithm used for estimating the state of a dynamic system based on noisy measurements. The goal of this TP was to implement a basic Kalman Filter for tracking a single object in a simple video scene.

### Implementation
- The first step was building the `KalmanFilter` structure. This included defining the state vector, measurement vector, and the necessary matrices (state transition, observation, process noise covariance, and measurement noise covariance). The state vector included the position and velocity of the object. No major difficulty occurred here.
- Implementing the prediction function was straightforward. The state was predicted using the state transition matrix.
- The update function, however, caused several issues. The main problem came from misunderstanding how the measurement coordinates were passed to the function, resulting in dimension mismatches due to squeezed arrays. After correcting the input shapes, the function behaved as expected.
- When integrating the filter into the tracking pipeline, I initialized it with the first detection and then called prediction and update for each subsequent frame. Initially, the predicted box moved along a straight line at the bottom of the screen. This was due to an indexing error in the update method. After fixing it, the tracker behaved correctly, although the prediction always seemed to lag behind the true position. I eventually discovered that I was passing the previous predicted state instead of the updated state to the next iteration. Fixing this resolved the issue.

### Results
The Kalman Filter tracker successfully tracks the object throughout the sequence. The predicted bounding boxes closely follows the true detections, demonstrating the effectiveness of the filter in a simple scenario.  
The red box represents the actual detection and the blue box is the Kalman prediction.

![Kalman Filter Tracking Results](./gifs/tp1.gif)

---

## TP2: Multi-Object Tracking using IoU-Based Data Association

IoU (Intersection over Union) is a metric used to evaluate the overlap between two bounding boxes. The goal of this TP was to implement a multi-object tracker using IoU-based data association along with the Hungarian algorithm. The detections were provided in a CSV file.

### Implementation
- To structure the problem, I created several classes:  
  1. `Point`: represents a 2D point with integer x and y coordinates.  
  2. `BoundingBox`: defined by its top-left and bottom-right corners.  
  3. `Track`: represents a tracked object, with an ID, a bounding box, and a counter for missed frames.  
  4. `TrackList`: originally intended to manage tracks but eventually unused in favor of a simple Python list.
- After loading the detections, I mapped them to bounding boxes and implemented the IoU function. This part was straightforward.
- For the first frame, each detection creates a new track.
- For later frames, I computed the IoU between each existing track and each detection to create an IoU matrix, then converted it to a cost matrix using `1 - IoU`.
- The Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) was used to find optimal associations. My main difficulty here was mixing track IDs, detection IDs, and Hungarian indices, leading to unstable track IDs. After realizing the Hungarian indices must be mapped back to the corresponding list indices, the associations became consistent.
- A threshold of 0.3 was used to discard weak associations. This means that any lower IoU values would not result in a match, and no Bounding Box would be displayed.
- The final associations were stored in `./ADL-Rundle-6/gt/tp2_det.txt`, with all detections associated with a matching ID.

### Results
The IoU-based tracker correctly handled multiple objects, maintaining coherent IDs during most of the sequence. ID swaps occasionally occurred when objects crossed paths, and some bounding boxes were shaking since the raw detections were used without any smoothing or prediction. Below is a visualization of the tracking results. Each bounding box is labeled with its corresponding track ID.

![IoU-based Multi-Object Tracking Results](./gifs/tp2.gif)

---

## TP3: Kalman Filter with IoU-Based Data Association for Multi-Object Tracking

This TP combined the previous two: a Kalman Filter was added to each track to improve prediction and overall stability.

### Implementation
- I refactored the code to make it cleaner. The cost matrix computation and frame-rendering logic were extracted into separate functions.
- I extended the `BoundingBox` class to be able to compute its own center, as well as shift the box based on the Kalman prediction of a new box center. Each `Track` now included its own Kalman Filter.
- The logic was non trivial and had me thinking for a long time, but I eventually isolated that I had to handle three cases:
  1. **Matched detections**: Matched detections were the first case I handled. These detections can be defined as those with a great IoU with an existing track. For these detections, I updated the corresponding Kalman Filter with the detection's center as the measurement. After updating the Kalman Filter, I obtained the predicted state and used it to update the bounding box of the track. The updated box was displayed on the frame with its ID, and the predicted box was stored as the track's bounding box for better IoU score at the next step. I did not encounter issues with this part.

  2. **Unmatched detections**: Unmatched Detections: Unmatched detections were the next case I handled, and also the trickiest one. These detections can be defined as those with a low IoU with all existing tracks, and can be considered as new objects entering the scene. For these detections, a new track is created and initialized with a new Kalman Filter with the detection's center as the initial state. The new track was added to the list of existing tracks, and its bounding box was displayed on the frame with a new ID. The main issue I encountered here was detecting which detections were unmatched. I initially tried to do this by checking the IoU values, but it was error-prone and led to incorrect associations quite often. The solution I opted for was to store all detections that were matched during the previous step, and then simply check which detections were not in this list. This approach worked much better, once I figured out a way to manage the track IDs in a better way, since my previous implementation did not allow to assign IDs to tracks the way I needed.

  3. **Unmatched tracks**: Unmatched tracks were the final case I handled. This was also the easiest one by far. The intended way to handle unmatched tracks is to increase their "missed frames" counter, and if this counter exceeds a certain threshold, the track is considered dead and removed from the list of existing tracks. But instead of doing this, I simply considered all tracks as unmatched at the beginning of the step, increasing all the missed counters by one. Then, during the matched detections step, I reset the counter for matched tracks back to zero. Finally, at the end of the loop, I removed all tracks with a missed frames counter exceeding a threshold and for thoses which did not reach this threshold, I displayed the prediction of the Kalman Filter on the frame, to infer the object's position. This approach worked well and simplified the logic significantly.


- Overall, the integration of the Kalman Filter into the IoU-based multi-object tracker improved the tracking performance, resulting in smoother trajectories and more consistent IDs for each object. Although, I noticed that some boxes would appear out of nowhere and go across the screen with high velocity. This was most likely due to detections of a weird object for a few frames, causing the Kalman Filter to predict a high velocity for the object (whether it actually existed or not). To fix this, I added a counter to the Track class that counts the total number of frames where a detection was matched with the track. This allowed to set up a threshold to assert that all predictions could only be displayed if the track had been matched at least a certain number of times, which guaranteed that the track was actually valid, and not some "phantom" track.

### Results
The integration of the Kalman Filter produced smoother trajectories and more consistent IDs. Although some phantom boxes moving across the screen still appeared in dense scenes, the overall result was noticeably better than TP2. Below is a visualization of the tracking results. Each bounding box is labeled with its corresponding track ID. When a bounding box is inferred, the word "Predicted" is displayed next to the ID.

![Kalman Filter with IoU-based Multi-Object Tracking Results](./gifs/tp3.gif)

---

## TP4: Deep Learning-Based Object Tracking with IoU and Kalman Filter

This TP was to go even further beyond the previous TPs by integrating a Deep Learning-based object detector into the tracking pipeline. The goal was to use a pre-trained object detection model to provide a similarity score for data association, instead of relying solely on the IoU metric. The Kalman Filter would still be used for predicting object positions between frames.

### Implementation
- The changes in the code were minimal. The only function I changed is the function that computes the cost matrix, which now also computes the cosine similarity for each pair of track and detection feature vectors. The cost matrix was then computed as a weighted sum of the IoU cost and the cosine similarity cost, with weights defined as hyperparameters ALPHA and BETA (set to 0.7 and 0.3, respectively).
- The feature extraction process required some preprocessing beforehand. First, each bounding box was cropped from the frame and resized to the input size expected by the model (64x128). The cropped image was then normalized using the mean and standard deviation values provided in the TP instructions. Finally, the preprocessed image was passed through the model to obtain the feature vector. I had several issues with this part, due to image boxes being out of bounds of the image, especially with predictions since these are inferred and fixed on the velocity of the object. Several fixed were required to prevent this. I clamped the box coordinates to always be positive and stay in bounds of the image. This was not a trivial task since I had implemented operators for Points, which also had to be changed. Another issue I faced were boxes with zero area (corners sharing an x or y coordinate), which caused errors during resizing. To fix this, I added a check to ensure that the width and height of the box were at least 1 pixel before cropping and resizing (this would only occur on the edges of the picture, and most likely apply to only objects which left the scene).
- At this point, the code was working but it was extremely slow. I ended up realizing that calling a forward pass on the model for each couple of box, for each frame, was extremely inefficient. To fix this, I decided to precompute all feature vectors for all detections in the current frame before computing the similarity matrix in one call to the cosine similarity function from sklearn. This change drastically improved the speed of the tracker.


### Results
Combining appearance features with IoU and Kalman prediction improved data association, reducing ID switches and producing more stable tracking results in crowded scenes. Below is a visualization of the tracking results. Each bounding box is labeled with its corresponding track ID. When a bounding box is inferred, the word "Predicted" is displayed next to the ID.

![Deep Learning-based Multi-Object Tracking Results](./gifs/tp4.gif)

---

# Conclusion

These TPs were a valuable opportunity to explore different object-tracking approaches. Starting from a basic Kalman Filter for single-object tracking, the project evolved into multi-object tracking with IoU-based association and finally a more advanced system integrating deep-learning appearance features.

My favorite TP was TP3, as it already felt like a robust and well-balanced tracking system, offering smooth predictions and stable IDs. TP4 was interesting as well, but the performance gain was smaller than expected, and the implementation was more straightforward thanks to the work done in TP3.

Overall, these TPs were challenging but rewarding, and they helped me build a solid understanding of object tracking techniques and their practical implementation.