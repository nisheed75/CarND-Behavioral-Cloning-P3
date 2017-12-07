# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data on good driving behaviour
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
1.1. The get_training_data function takes the path walks the directories and looks for the driving log file. In my case, i made a few attempts to capture data, so I instruct the function to only look for version 2 of my log data. This function returns the path to the centre, left, right images and the measurements.
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
    If the file includes headers, pass `skipHeader=True`.
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

I used the Architecture recommended in lecture 11. nVidia has developed a deep learning model called PilotNet that derives the necessary domain knowledge by observing human drives. The image below shows the model architecture.


![alt text][image16]

This model consists of 9 layers including:
1. A normalization layer 
1. 5 Convolutional layers
1. 3 Fully connected layers 

The first layer of the model performs image normalization. This is statically defined and doesn't change during the learning process.

Through the convolution layers, the model performs feature extraction. Nvidia has experimented with this and found the best configuration for the convolution layers. The first 3 layers use a stride of 2x2 and 5x5 kernel and a non-stride convolution with a 3 x 3 in the last 2 layers.

It is then followed by 5 fully connected layers leading to the output control value. The fully connected layers are designed to be the controller for steering.  


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers to reduce overfitting . 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of centre lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I chose the nVidia model as this model is a tried and test model and seems to be a greate start for completing this project as explained in lecture 14. 

To gauge how well the model was working, I split my image and steering angle data into a training and validation set.

To ensure the model wsn't overfitting I added droput to the model. 

The final step was to run the simulator to see how well the car was driving around track one. I got lucky as the first model i trained allowed my car to drive around the track with steering off. There were time when there went clsoe to the edge but is gracefully steerd back to the centre. 

At the end of the process, the vehicle can drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 107-144) is described in detail in section 1 above. The following code shows the layers of the model:

```python
def create_pre_processing_model():
    """
    Creates a model with the initial pre-processing layers.
    """
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    return model

#Code borrowed from Lecture 14 updated the code to use the new Keras API
def nvidia_model():
    """
    Creates nVidia Autonomous Neural Network Model as described in Lecture 14
    """
    model = create_pre_processing_model()
    model.add(Conv2D(24, (5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(36, (5,5), strides=(2,2), activation='relu'))
    model.add(Conv2D(48, (5,5), strides=(2,2), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

```

Here is a visualization of the architecture 

![alt text][image16]

#### 3. Creation of the Training Set & Training Process

To gather training data I first took a lap around the track without recording my driving behaviour. This was done so I could get a feel for controlling the car. Once I was comfortable doing this, I recorded my driving for one complete lap where I kept the car on the track throughout the lap.

The driver log data produced by the recorded had images for left, centre and right images with measurements taken for steering angle, throttle, brake & speed. Here are sample images of the left centre and right for snapshots of the track.

##### Example 1
![alt text][image1]![alt text][image2]![alt text][image3]

##### Example 2
![alt text][image4]![alt text][image5]![alt text][image6]

##### Example 3
![alt text][image7]![alt text][image8]![alt text][image9]

##### Example 4
![alt text][image10]![alt text][image11]![alt text][image12]

##### Example 5
![alt text][image13]![alt text][image14]![alt text][image15]

Once I had the data collected I preprocessed my data to get a single data set that showed the car the steering angle for being in the centre of the road. I also used the left and tight images to generate data to teach the car how to drive back to the centre. Since the data collection for the left and right did not have steering angles for going back to centre, i used a correction factor to adjust the steering angle to get the car to steer back to centre.


I finally randomly shuffled the data set and used the following code to randomly break my data into train, validation and test datasets.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by my loss data for training and validation below:
```
Epoch 2/5
663/663 [==============================] - 288s - loss: 0.0158 - val_loss: 0.0078
Epoch 2/5
663/663 [==============================] - 285s - loss: 0.0055 - val_loss: 0.0048
Epoch 3/5
663/663 [==============================] - 286s - loss: 0.0031 - val_loss: 0.0036
Epoch 4/5
663/663 [==============================] - 271s - loss: 0.0022 - val_loss: 0.0037
Epoch 5/5
663/663 [==============================] - 275s - loss: 0.0017 - val_loss: 0.0029
dict_keys(['loss', 'val_loss'])
Loss
[0.015838943200306509, 0.0055239118810277432, 0.0031207919476804917, 0.0022257506585183335, 0.0017283360391687826]
Validation Loss
[0.0078371490974650203, 0.0048086696523388711, 0.0035702709243861271, 0.0037371346694052322, 0.0028714236875847601]