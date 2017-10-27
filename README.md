# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/histogram.png "Histogram"
[image2]: ./images/before_grayscale.png "Grayscaling Sample Image"
[image11]: ./images/after_grayscale.png "Grayscaling Sample Image"
[image3]: ./images/before_norm.png "Normalization"
[image12]: ./images/after_norm.png "Normalization"
[image4]: ./german_traffic_signs/double_curve_21.jpg "Traffic Sign 1"
[image5]: ./german_traffic_signs/general_caution_18.jpg "Traffic Sign 2"
[image6]: ./german_traffic_signs/no_entry_17.jpg "Traffic Sign 3"
[image7]: ./german_traffic_signs/no_passing_9.jpg "Traffic Sign 4"
[image8]: ./german_traffic_signs/pedestrians_crossing_27.jpg "Traffic Sign 5"
[image9]: ./german_traffic_signs/speed_limit_80_5.jpg "Traffic Sign 6"
[image10]: ./german_traffic_signs/turn_right_ahead_33.jpg "Traffic Sign 7"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/elok/CarND-Traffic-Sign-Classifier-Project)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799
* The size of the validation set is 4,410
* The size of test set is 12,630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. The following bar chart shows the distribution of the number of each sign. For example, in the X_train dataset, there are about 2000 number 2 signs. Blue is the training data set. Red is the test data set and green is the validation data set.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale. The idea is to simplify the color channels from 3 to 1. It showed a bit of improvement but not alot. Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]
![alt text][image11]

As a last step, I normalized the image data to keep the mean at zero and the variance at 1 by subtracting by 127.5 and dividing by 255.

I did not generate additional data. It wasn't clear to me where to generate it from and how it would benefit the training workflow.

Here is an example of an original image and an augmented image:

![alt text][image3]
![alt text][image12]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   	     			| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Dropout				|												|
| Max pooling	      	| 2x2 stride, outputs 14x14x6 		     		|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 10x10x16 	|
| RELU					|												|
| Dropout				|												|
| Max pooling	      	| 2x2 stride, outputs 5x5x16 			    	|
| Flatten               | outputs 400                                   |
| Fully connected		| outputs 120       							|
| RELU					|												|
| Dropout				|												|
| Fully connected		| output 84     								|
| RELU					|												|
| Dropout				|												|
| Fully connected		| output 43        								|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I experimented with alot of parameters. It was very difficult to keep track. I would make a list and try to keep track of all the parameters I tweaked and the training results.

* 87% original
* 85% with equalize_hist
* 84% with learning rate 0.0004
* 90% with learning rate 0.0004 and 20 epoch
* 89% with learning rate 0.001 and 20 epoch
* 92% with learning rate 0.0004 and 50 epoch
* 90% with learning rate 0.0004 and 50 epoch and grayscale
* 89% 50 epoch, grayscale, and 128 normalization
* 91% 50 epoch, grayscale, and (/ 255) normalization, 0.0004
* 92.6% 50 epoch, grayscale, and (/ 255) normalization, 0.001
* 89.1% 50 epoch, grayscale, and (/ 255) normalization, 0.01
* 84.4% 50 epoch, grayscale, and (/ 255) normalization, 0.0001
* 94% 50 epoch, grayscale, and (/ 255) normalization, 0.002
* 93.8% 50 epoch, grayscale, and (/ 255) normalization, 0.003
* 93% 50 epoch, grayscale, and (/ 255) normalization, 0.004
* 91.3% 50 epoch, grayscale, and (/ 255) normalization, 0.002, dropout 0.5
* 91.7%  200 epoch, grayscale, and (/ 255) normalization, 0.002, dropout 0.5
* 93.8% 30 epoch, grayscale, and (/ 255) normalization, 0.002, dropout 0.3
* 93.7% 50 epoch, grayscale, and (/ 255) normalization, 0.002, dropout 0.5 (dropoutx1)
* 95.8% 50 epoch, grayscale, and (/ 255) normalization, 0.002, dropout 0.5 (dropout x3)
* 97% 100 epoch, grayscale, and (/ 255) normalization, 0.002, dropout 0.5 (dropout x3)
* 96.2% 200 epoch, grayscale, and (/ 255) normalization, 0.002, dropout 0.5 (dropout x3)
* 97.2% 80 epoch, grayscale, and (/ 255) normalization, 0.002, dropout 0.5 (dropout x3)
* 95.7% 60 epoch, grayscale, and (/ 255) normalization, 0.002, dropout 0.5 (dropout x3)
* 96.8% 120 epoch, grayscale, and (/ 255) normalization, 0.002, dropout 0.5 (dropout x3)
* 96.5% 80 epoch, grayscale, and (/ 255) normalization, 0.002, dropout 0.5 (dropout x3) (retry)


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.1%
* validation set accuracy of 95.1%
* test set accuracy of 93.8%

The LeNet architecture was chosen to train the dataset and the hyperparameters were tweaked using an iterative approach. LeNet architecture is an image classification system that would transfer very nicely to our problem of classifying traffic signs. Out of the box, the results were not that great with an accuracy of 87%. Adding preprocessing and tweaking some initial parameters did bump the accuracy over 90%.

I believe the model was overfitting and decided to add dropout after every RELU. This bumped up the accuracy considerably. The parameters I tuned were learning rate, epoch, batch_size, and keep probability. My approach to tuning the parameters were to first use extreme numbers (high and low) to see what affect it had on the model. I would then try suggested starting numbers that were discussed in the course and then start experimenting from that point in a direction that would yield better results.

I believe adding the dropout to my model was important to keep it from overfitting but at the same time, I noticed that I had to adjust the epoch at the same time. Using dropout with a low epoch did not produce good results. If you were to use dropout, a pretty high number of epochs was necessary to train the data set properly. In my test, I found 80 to be a good number.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8] ![alt text][image9]
![alt text][image10]

In general, I don't think any of the images that I've chosen would be difficult to classify but I can understand why some might. For example, if it was cloudy and the image is dark, it might be hard to identify the sign. Shadows, reflections, and rain might also affect how the sign will look. I tried to choose bright images but now that I look back at it, the preprocessing steps might have darkened the images and might need some brightening.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Double Curve     		| Children crossing   							| 
| General caution     	| General caution 								|
| No entry				| No entry										|
| No passing	      	| No passing					 				|
| Pedestrians Crossing	| General caution      							|
| Speed Limit 80		| Speed limit (50km/h)      					|
| Turn right ahead		| Turn right ahead      						|

The model was able to correctly guess 4 of the 7 traffic signs, which gives an accuracy of 57%. This accuracy is pretty low. I would expect it to have done better.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 19th cell of the Ipython notebook.

# 1. Double Curve (wrong)
For the first image, the model is relatively sure that this is a Children Crossing sign (probability of 0.88) but the image is a double curve. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 8.81066263e-01    	| Children crossing                             | 
| 8.20478648e-02    	| Dangerous curve to the right					|
| 1.34377247e-02		| Right-of-way at the next intersection 		|
| 1.16430800e-02		| Beware of ice/snow 			 				|
| 6.69767335e-03	    | Slippery road      							|

# 2. General Caution (correct)
For the second image, the model is relatively sure that this is a General Caution sign (probability of 0.99) and it is correct. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99581039e-01      	| General caution								| 
| 3.85274470e-04        | Traffic signals   							|
| 3.37743732e-05		| Pedestrians									|
| 1.32089326e-10		| Road narrows on the right     				|
| 9.81628320e-12	    | Right-of-way at the next intersection 		|

# 3. No Entry (correct)
For the third image, the model is relatively sure that this is a No Entry sign (probability of 0.99) and it is correct. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.96406615e-01    	| No entry   									| 
| 3.02194571e-03    	| Stop 							    			|
| 4.59565083e-04	    | Turn left ahead								|
| 5.30324724e-05	 	| Turn right ahead								|
| 2.66978532e-05	    | Roundabout mandatory      					|

# 4. No Passing (correct)
For the fourth image, the model is relatively not sure that this is a No Passing sign (probability of 0.73) even though it is correct. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 7.32200146e-01      	| No passing   									| 
| 1.05444029e-01        | Slippery road 								|
| 6.79857880e-02		| End of no passing 							|
| 5.51116243e-02		| Vehicles over 3.5 metric tons prohibited 		|
| 1.76423192e-02	    | Children crossing    							|

# 5. Pedestrians Crossing (wrong)
For the fifth image, the model is relatively not sure that this is a General Caution sign (probability of 0.53) and indeed it is wrong. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 5.34613550e-01    	| General caution  								| 
| 3.24179828e-01    	| Traffic signals  								|
| 4.62946296e-02		| Pedestrians		    						|
| 1.81003790e-02		| Road narrows on the right		 				|
| 1.43758226e-02	    | Dangerous curve to the right  				|

# 6. Speed Limit 80 (wrong)
For the sixth image, the model is relatively not sure that this is a Speed Limit 50km/h sign (probability of 0.69) and the image is a Speed Limit 80km/h. I can understand how a 5 can be interpreted as an 8. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 6.96982980e-01    	| Speed limit (50km/h)                      	| 
| 1.33007288e-01        | Speed limit (20km/h) 							|
| 1.15418367e-01		| Speed limit (80km/h)							|
| 3.51988822e-02		| Speed limit (70km/h)					 		|
| 9.10976343e-03	    | Speed limit (100km/h)      				    |

# 7. Turn Right Ahead (correct)
For the seventh image, the model is relatively sure that this is a Turn Right Ahead sign (probability of 0.99) and it is indeed correct. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99830961e-01    	| Turn right ahead   				        	| 
| 1.67640334e-04    	| Ahead only 									|
| 7.38871051e-07	    | Road work										|
| 2.30277521e-07	  	| Right-of-way at the next intersection			|
| 1.44437493e-07	    | Keep left      			    				|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

I'm not sure if I did this step completely correct. From what I can tell, the network uses lines or edge detection as the major features for classification.
