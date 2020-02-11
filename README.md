# CNN-vis
CNN-vis is a flexible Python software which is executable from the command line. Its main purpose is to visualize why a Convolutional Neural Network (CNN) is doing what it does. At the moment, four main visualizations are supported: Convolutional Kernel visualization, Activation Maximization technique, Saliency maps and Gradient Class Activation Maps (Grad-CAM), from the [Keras-vis library](https://raghakot.github.io/keras-vis/) of Github user Raghakot, along with other small utilities.

This software was developed for my Bioinformatics project of 7.5 ECTS with Fernando Racimo's group, at the Univeristy of Copenhagen, with title 'Understanding a CNN to predict Adaptive Introgression'.

Manual to run CNN-vis from the console:

Operation | Flag | Comments | Required | Type
------------ | ------------- | ------------- | -------------  | -------------
Function | function | Visualization technique or utility to perform. Choices are: *activation-max*, *saliency*, *grad-cam*, *filter-weights*, *model-summary*, or *get-labels* | Yes | String
ConvNet model | model | Path to model file, stored as a .h5 Python format | Yes | String
Data | -d or --data | Path to data containing image matrices X and labels y as a pickle package in .pkl format | No | String
Layer | -l or --layer | Name of the desired Conv layer in the ConvNet model to visualize | No | String
Image indeces | -im or --images | Indeces to the selected images for saliency and grad-CAM visualization | No | List of integers
Backpropagation modifier | -bm or --backprop | Modifier for the backpropagation step. Choices are: *None (vanilla)*, *guided*, *rectified*, *deconv*, or *relu*. Guided backpropagation is recommended and selected by default | No | String


## Functions
- **activation-max:** Convolutional filter visualization via Activation Maximization technique ([Dumitru Erhan et al.](https://pdfs.semanticscholar.org/65d9/94fb778a8d9e0f632659fb33a082949a50d3.pdf)). This method is used to get an insight into what input motifs activate a particular filter. When applied in the final Dense layer, it's possible to visualize which is the input image that the ConvNet expects in order to maximize a certain output class.

- **saliency:** Attention visualization technique with discriminative behavior based on estimating the gradient of an output category with respect to the input image. ([Karen Simonyan et al.](https://arxiv.org/abs/1312.6034)). These attention maps help to visualize which regions of the input image draw the most attention from the ConvNet model.

- **grad-cam** Attention visualization technique with discriminative behavior based on estimating the gradient of an output category with respect to the input image. ([Ramprasaath R. Selvaraju et al.](https://arxiv.org/abs/1610.02391)). It differs with the saliency maps in the sense that it uses the output of the previous Conv layer before the specified one.

- **filter-weights:** Visualization of the learned weights from all the filters in the provided Conv layer.

- **model-summary:** Prints the information of the provided CNN architecture to the user.

- **filter-weights:** Prints the label and the index of the input images provided as data.

## Library requirements:
Tensorflow 2.0.*

Keras-vis 0.4.1


## Example of terminal command to run the code:
python3 cnn-vis.py saliency data/my_model.h5 --data data/my_image_data.pkl --layer output --images 0 3 5 10


