# DevNetworkHackathon
This repository contains my attempt for DevNetwork hackathon.

The project contains three steps - 



🔹 Speech Therapy – Lip Movement Analysis
In traditional speech therapy, children are guided to pronounce specific words or phrases, while therapists observe and evaluate their articulation. To digitize this process:

Developed a system to compare a reference video (recorded by a parent or therapist) with the child’s attempt to repeat the word.

Integrated the shape_predictor_68_face_landmarks.dat model to accurately track facial landmarks and analyze lip movements.

Used Euclidean distance to measure the variation in lip positions across video frames, thereby quantifying pronunciation accuracy.

Visualized the comparison with side-by-side video playback and dynamic graphs with score to detect even minor articulation 
differences.

1) Comparsion graph between lip distance for both videos
![App Screenshot](LipMovementComparison/Output/graph.png)

2) Similarity score between lip movement of two videos 
 ![App Screenshot](LipMovementComparison/Output/similarityScore.png)

3) Comparison video  
![App Screenshot](LipMovementComparison/Output/sideToSideVideoComparisonWithRealTimeScore.png)




🔹 Motor Therapy – Pose Estimation
Motor therapy often involves mimicking physical activities demonstrated by a therapist. My solution automates this assessment:

Utilized Detectron2’s Mask R-CNN to perform pose estimation and identify key body joints.

Recorded and compared the child’s actions with ideal movement references.

Calculated similarity scores to quantify motion accuracy.

Presented results using video overlays and visual graphs, enabling therapists to assess how well the child replicates movements.

1) Key Points of Video 1 in the form of CSV file
![App Screenshot](PoseEstimation/Output/PoseEstimationKeypointExtractionIndividualVideo1.png)

2) Key Points of Video 2 in the form of CSV file
![App Screenshot](PoseEstimation/Output/PoseEstimationKeypointExtractionIndividualVideo2.png)

3) Key Points of combined summary of both videos in the form of CSV file
![App Screenshot](PoseEstimation/Output/PoseEstimationKeypointExtractionSummaryBothVideos.png)

4)Cosine similarity across each frames in both videos
![App Screenshot](PoseEstimation/Output/PoseEstimationGraph.png)

5) Similar Pose comparison across each individual frame in both videos
![App Screenshot](PoseEstimation/Output/PoseEstimationIndividualPoseComparison.png)

6) Similar Pose comparison in skeleton across each individual frame in both videos
![App Screenshot](PoseEstimation/Output/PoseEstimationIndividualSkeletonPoseComparison.png)

7) Comparison of both the videos side by side  
![App Screenshot](PoseEstimation/Output/PoseEstimationSideBySide.png)





🔹 Interpretation Therapy – Visual Understanding through Drawing

Many children with developmental disorders struggle with connecting words to visuals. This module evaluates their ability to interpret verbal prompts:

Used the QuickDraw dataset and Python API to simulate real-time drawing tasks.

Displayed a simple word prompt (e.g., “cat”) and prompted the child to draw the object.

Compared the child’s drawing to multiple reference samples from QuickDraw.

Measured semantic alignment between the prompt and drawing, offering insight into cognitive and interpretive skills.

1) First screen where category for the scribble is chosen
   
![App Screenshot](QuickDrawScribble/Output/QuickDrawScribbleChooseCategory.png)

2) User scribbles for the category
![App Screenshot](QuickDrawScribble/Output/QuickDrawScribbleDrawing.png)

3) User scribble is being compared with the drawing in dataset
![App Screenshot](QuickDrawScribble/QuickDrawScribbleResult.png)

Remark - Every folder contains the python code of each step. There is a sub-folder 'Output' inside each folder which contains screenshots, graph results, videos and inputs of the code. Please go through that. For any questions or queries, please email me @ pankhuri.nyu@gmail.com

The demo of the video is over here - https://www.youtube.com/watch?feature=shared&v=BwIYqOn4sgs
