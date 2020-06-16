[![Udacity Computer Vision Nanodegree](http://tugan0329.bitbucket.io/imgs/github/cvnd.svg)](https://www.udacity.com/course/computer-vision-nanodegree--nd891)
# Facial Keypoint Detection
## Objective

This project aimed to build a software that can take in a given image of a person and automatically predict certain key-points of the persons face.

Therefore a CNN that is able to predict the key-points in an given input image had to be created.
Furthermore the preprocessing of the data as well as the training of the build network structure are main parts of the project.

## Results 

Below a pipline as executed in notebook three is shown. First the face in the given image is detected using openCV. After thast, the network detects the key points on the given face. In the last steps the key points are used to put Mona Lisa some sunglasses on.


![animated1](images/mona_lisa_pipeline.gif)
