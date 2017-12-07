# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)
[image1]: ./examples/left/left_2017_12_03_20_10_06_115.jpg "Example 1 Left"
[image2]: ./examples/center/center_2017_12_03_20_10_06_115.jpg "Example 1 Center"
[image3]: ./examples/right/right_2017_12_03_20_10_06_115.jpg "Example 2 Right"
[image4]: /examples/left/left_2017_12_03_20_10_08_363.jpg "Example 2 Left"
[image5]: /examples/center/center_2017_12_03_20_10_08_363.jpg "Example 2 Center"
[image6]: /examples/right/right_2017_12_03_20_10_08_363.jpg "Example 2 Right"
[image7]: /examples/left/left_2017_12_03_20_10_08_901.jpg "Example 3 Left"
[image8]: /examples/center/center_2017_12_03_20_10_08_901.jpg "Example 3 Center"
[image9]: /examples/right/right_2017_12_03_20_10_08_901.jpg "Example 3 Right"
[image10]: /examples/left/left_2017_12_03_20_10_13_020.jpg "Example 4 Left"
[image11]: /examples/center/center_2017_12_03_20_10_13_020.jpg "Example 4 Center"
[image12]: /examples/right/right_2017_12_03_20_10_13_020.jpg "Example 4 Right"
[image13]: /examples/left/left_2017_12_03_20_11_30_532.jpg "Example 5 Left"
[image14]: /examples/center/center_2017_12_03_20_11_30_532.jpg "Example 5 Center"
[image15]: /examples/right/right_2017_12_03_20_11_30_532.jpg "Example 5 Right"
[image16]: /examples/nVidia_model.png "nVidia Architecture"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. I also explain the data pipeline below.

My pipeline was as follow:

1. I get the training data by passing in the location of the data. 
```python
	center_paths, left_paths, right_paths, measurements = get_training_data('data\sim_data')
```
1.1. The get_training_data function takes the path walks the directores and looks for the driving log file. In my case i made a few attempts to capture data so i instruct the function to omly look for version 2 of my log data. This function returns the path to the center, left, right images and the measurements.
```python
def get_training_data(dataPath):
    """
    Finds all the images needed for training on the path `dataPath`.
    Returns `([center_paths], [leftPath], [rightPath], [measurement])`
    """
    directories = [x[0] for x in os.walk(dataPath)]
    dataDirectories = list(filter(lambda directory: os.path.isfile(directory + '\\driving_log_2.csv'), directories))
    center_total = []
    left_total = []
    right_total = []
    measurement_total = []
    for directory in dataDirectories:
        lines = get_log_data(directory)
        center = []
        left = []
        right = []
        measurements = []
        for line in lines:
            measurements.append(float(line[3]))
            source_path =  line[0].strip()
            filename = source_path.split('\\')[-1]
            center.append(directory + '\\IMG\\' +filename)
            source_path =  line[1].strip()
            filename = source_path.split('\\')[-1]
            left.append(directory + '\\IMG\\' + filename)
            source_path =  line[2].strip()
            filename = source_path.split('\\')[-1]
            right.append(directory + '\\IMG\\' + filename)
        center_total.extend(center)
        left_total.extend(left)
        right_total.extend(right)
        measurement_total.extend(measurements)

    return (center_total, left_total, right_total, measurement_total)	
```
1.1. The get_training data uses the get_log_data function to read each line in the log data file.

```python
def get_log_data(dataPath, skipHeader=False):
    """
    Returns the lines from a driving log with base directory `dataPath`.
    If the file include headers, pass `skipHeader=True`.
    """
    lines = []
    with open(dataPath + '\\driving_log_2.csv') as csvFile:
        reader = csv.reader(csvFile)
        if skipHeader:
            next(reader, None)
        for line in reader:
            lines.append(line)
    return lines
```
1. Once the data is read I then merge the center, left and right data into a single teianing set. i adjust the measurement for the left and right image by a corrextion factor to help the car steer back to the cneter. The code that does this is below:
```python
image_paths, measurements = merge_images(center_paths, left_paths, right_paths, measurements, 0.2)

def merge_images(center, left, right, measurement, correction):
    """
    Combine the image paths from `center`, `left` and `right` using the correction factor `correction`
    Returns ([image_paths], [measurements])
    """
    image_paths = []
    image_paths.extend(center)
    image_paths.extend(left)
    image_paths.extend(right)
    measurements = []
    measurements.extend(measurement)
    measurements.extend([x + correction for x in measurement])
    measurements.extend([x - correction for x in measurement])
    return (image_paths, measurements)
```
### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used the Architecture recommended in lecture 11. nVidia has develop a deep learning model called PilotNet that derives the neccesary domain knowledge by observing human drives. The image below shows the model architecture.


![alt text][image16]

This model consist of 9 layers including:
1. A normalization layer 
1. 5 Convolutional layers
1. 3 Fully connected layers 

The first layer of the model performs image normlization. This is statically defined and doesn't change during the learning process.

Through the convolution layers the model performs feature extraction. nVidia has expereiment with this and found the best confiuration for the convolution layers. The first 3 layers use a stride of 2x2 and 5x5 kernel and a non stride convolution with a 3 x 3 in the last 2 layers.

It is then followed by 5 fully connected layers leading to the output control value. The fully connected layers are designed to be the controller for steering.  


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
