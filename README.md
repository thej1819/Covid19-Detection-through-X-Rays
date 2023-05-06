# Covid19-Detection-through-X-Rays
Project Description:

 COVID-19 (coronavirus disease 2019) is an infectious disease caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), which is a strain of coronavirus. 
 The disease was officially announced as a pandemic by the World Health Organization(WHO) on 11 March 2020. Given spikes in new COVID-19 cases and the re-opening of daily activities around the world, 
 the demand for curbing the pandemic is to be more emphasized. Medical images and artificial intelligence (AI) have been found useful for rapid assessment to provide treatment of COVID-19 infected patients.
 The PCR test may take several hours to become available, information revealed from the chest X-ray plays an important role in a rapid clinical assessment. This means if the clinical condition and the chest X-ray are normal, 
 the patient is sent home while awaiting the results of the etiological test. But if the X-ray shows pathological findings, the suspected patient will be admitted to the hospital for close monitoring. Chest X-ray data have
 been found to be very promising for assessing COVID-19 patients, especially for resolving emergency-department and urgent-care-center overcapacity. Deep-learning (DL) methods in artificial intelligence (AI) play a dominant 
 role as high-performance classifiers in the detection of the disease using chest X-rays.

 

One of the biggest challenges following the Covid-19 pandemic is the detection of the disease in patients. To address this challenge we have been using the Deep Learning Algorithm to build an image recognition model that can 
detect the presence of Covid-19 from an X-Ray or CT-Scan image of a patient's lungs.

Transfer learning has become one of the most common techniques that have achieved better performance in many areas, especially in medical image analysis and classification. We used Transfer Learning techniques like
Inception V3,Resnet50,Xception V3 that is more widely used as a transfer learning method in medical image analysis and are highly effective.


# Data Collection
  There are many popular open sources for collecting the data. Eg: kaggle.com, UCI repository, etc.
  Download The Dataset
Collect images of Covid-19 Chest X-ray images then organized them into subdirectories based on their respective names as shown in the project structure. Create folders of types of Covid-19 that need to be recognized
.
In this project, we have collected images of 2 types of Covid-19 images Covid-19 positive and Covid-19 negative, and they are saved in the respective sub directories with their respective names.


# Create Training And Testing Dataset
To build a DL model we have to split training and testing data into two separate folders. But In the project dataset folder training and testing folders are presented. So, in this case, we just have to assign a variable and pass the folder path to it.

Four different transfer learning models are used in our project and the best model (Xception) is selected.

The image input size of xception model is 299, 299.


# Image Preprocessing
In this milestone, we will be improving the image data that suppresses unwilling distortions or enhances some image features important for further processing, 
although performing some geometric transformations of images like rotation, scaling, translation, etc.

 
# Configure ImageDataGenerator Class

ImageDataGenerator class is instantiated and the configuration for the types of data augmentation

There are five main types of data augmentation techniques for image data; specifically:

        Image shifts via the width_shift_range and height_shift_range arguments.
        The image flips via the horizontal_flip and vertical_flip arguments.
        Image rotations via the rotation_range argument
        Image brightness via the brightness_range argument.
        Image zoom via the zoom_range argument.
        
An instance of the ImageDataGenerator class can be constructed for train and testing. 


# Apply ImageDataGenerator Functionality To Train Set And Test Set

Let us apply ImageDataGenerator functionality to the Train set and Test set by using the following code. For Training set using flow_from_directory function.

This function will return batches of images from the subdirectories

Arguments:

        directory: Directory where the data is located. If labels are "inferred", it should contain subdirectories, each containing images for a class. Otherwise,                    the directory structure is ignored.
        batch_size: Size of the batches of data which is  32.
        target_size: Size to resize images after they are read from disk. 
        class_mode:

                   -  ‘int': means that the labels are encoded as integers (e.g. for sparse_categorical_crossentropy loss).

                   - 'categorical' means that the labels are encoded as a categorical vector (e.g. for categorical_crossentropy loss).

                   - 'binary' means that the labels (there can be only 2) are encoded as float32 scalars with values 0 or 1 (e.g. for binary_crossentropy).

                   - None (no labels).
 
 # Model Building
## Pre-Trained CNN Model As A Feature Extractor
For one of the models, we will use it as a simple feature extractor by freezing all the five convolution blocks to make sure their weights don’t get updated after each epoch as we train our own model.

Here, we have considered images of dimension (299,299,3).

 Also, we have assigned include_top = False because we are using the convolution layer for features extraction and want to train a fully connected layer for our images classification(since it is not the part of Imagenet dataset)

Flatten layer flattens the input. Does not affect the batch size.


# Adding Dense Layers
A dense layer is a deeply connected neural network layer. It is the most common and frequently used layer.

Let us create a model object named model with inputs as xception. input and output as dense layer.
The number of neurons in the Dense layer is the same as the number of classes in the training set. The neurons in the last Dense layer, use softmax activation to convert their outputs into respective probabilities.

Understanding the model is a very important phase to properly using it for training and prediction purposes. Keras provides a simple method, a summary to get the full information about the model and its layers.
  
# Configure The Learning Process
The compilation is the final step in creating a model. Once the compilation is done, we can move on to the training phase. The loss function is used to find errors or deviations in the learning process. Keras requires a loss function during the model compilation process.

Optimization is an important process that optimizes the input weights by comparing the prediction and the loss function. Here we are using adam optimizer

Metrics are used to evaluate the performance of your model. It is similar to the loss function, but not used in the training process.  
  
  
# Train The Model
 

Now, let us train our model with our image dataset. The model is trained for 25 epochs and after every epoch, the current model state is saved if the model has the least loss encountered till that time. We can see that the training loss decreases in almost every epoch till 10 epochs and probably there is further scope to improve the model.

fit_generator functions used to train a deep learning neural network
 
Arguments:

-        steps_per_epoch: it specifies the total number of steps taken from the generator as soon as one epoch is finished and the next epoch has started. We can calculate the value of     steps_per_epoch as the total number of samples in your dataset divided by the batch size.

-        Epochs: an integer and number of epochs we want to train our model for.

-        validation_data can be either:

                      - an inputs and targets list

                      - a generator

                      - inputs, targets, and sample_weights list which can be used to evaluate

                        the loss and metrics for any model after any epoch has ended.

-        validation_steps: only if the validation_data is a generator then only this argument

can be used. It specifies the total number of steps taken from the generator before it is

stopped at every epoch and its value is calculated as the total number of validation data points

in your dataset divided by the validation batch size.
  
  
  
 # Save The Model
The model is saved with .h5 extension as follows

An H5 file is a data file saved in the Hierarchical Data Format (HDF). It contains multidimensional arrays of scientific data.
  
  
  
  
# Run The Application

-          Open the Command prompt from the start menu.

-          Navigate to the folder where your Python script is.

-          Now type the “python app.py” command.

-          Navigate to the localhost where you can view your web page.

-          Click on the predict button from the top right corner, enter the inputs, click on the submit button, and see the result/prediction on the web.


The home page looks like this. When you click on the button “Drop in the image you want to validate!”,

you’ll be redirected to the predict section  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  






























