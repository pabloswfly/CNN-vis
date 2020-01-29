# CNN-vis
CNN-vis is a flexible software executable from the command line. It's main purpose is to visualize why a Convolutional Neural Network is doing what it does. Three main visualizations are supported: Activation Maximization technique, Saliency maps and Gradient Class Activation Maps (Grad-CAM), from the Keras-vis library (https://raghakot.github.io/keras-vis/), along with other small utilities.

This software was developed for my Bioinformatics project of 7.5 ECTS in the Univeristy of Copenhagen, with title 'Understanding a CNN in Keras to predict Adaptive Introgression'.

Manual to run CNN-vis from the console:

Function -f:
Visualization technique or utility to perform. Choices are: 
‘activation-max’, ‘saliency’, ‘grad-cam’, ‘filter-weights’, ‘model-summary’, or ‘get-labels’.

ConvNet model -m:
Path to model file, stored as a .h5 Python format.

Data -d:
Path to data containing image matrices X and labels y as a pickle package in .pkl format.

Layer -l:
Name of the desired Conv layer in the ConvNet model to visualize

Image indeces -im:
Indeces to the selected images for saliency and grad-CAM visualization
List of integers

# Example of terminal command to run the code:
cnn-vis.py -f grad-cam -m data/your_cnn_model.h5 -d data/test_data.pkl -l output -im 0 1 2 5 10 18 19 31 51 72

