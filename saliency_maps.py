import sys

sys.path.insert(0,'/home/pabswfly/downloads/keras-vis' )

import numpy as np
import matplotlib.pyplot as plt
from vis.utils import utils
from vis.visualization import visualize_saliency
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import pickle
from mpl_toolkits.axes_grid1 import ImageGrid


def visualize_images(images, labels=None):
    """Plot a given picture"""

    fig = plt.figure(figsize=(8,3))

    grid = ImageGrid(fig, 111, nrows_ncols=(1, len(images)), axes_pad=0.15,
                 share_all=True, cbar_location="right", cbar_mode="single",
                 cbar_size="7%", cbar_pad=0.15)

    for i, im in enumerate(images):

        plot = grid[i].imshow(np.squeeze(im))

        if labels:
            grid[i].set_title(labels[i])

    grid[-1].cax.colorbar(plot)

    plt.title('Input images')
    grid[0].set_yticks([3, 35, 67])
    grid[0].set_yticklabels(['Neandertal', 'European', 'African'])
    plt.show()


def swap_function_to_linear(model, layer_name):
    """Given a model and a layer name, swaps the activation function of the layer for a linear one.
    Output: Returns the model with updated linear activation function"""

    layer = utils.find_layer_idx(model, layer_name)
    model.layers[layer].activation = keras.activations.linear

    model = utils.apply_modifications(model)

    return model



def plot_saliency(model, layer_name, images, backprop_mods = None, grad_mod = 'absolute', labels = None):

    layer = utils.find_layer_idx(model, layer_name)

    for modifier in backprop_mods:

        fig = plt.figure(figsize=(8, 3))

        grid = ImageGrid(fig, 111, nrows_ncols=(1, len(images)), axes_pad=0.15,
                         share_all=True, cbar_location="right", cbar_mode="single",
                         cbar_size="7%", cbar_pad=0.15)

        plt.suptitle(modifier)

        for i, im in enumerate(images):

            grads = visualize_saliency(model, layer, filter_indices=0, seed_input=im,
                                       backprop_modifier=modifier, grad_modifier= grad_mod)

            # Lets overlay the heatmap onto original image.
            plot = grid[i].imshow(grads, cmap='hot')

            if labels:
                grid[i].set_title(labels[i])

        grid[-1].cax.colorbar(plot)
        grid[0].set_yticks([3, 35, 67])
        grid[0].set_yticklabels(['Neandertal', 'European', 'African'])

    plt.show()



# opening the data. (1 == adaptive introgression, 0 otherwise)
with open("test_data_32x32.pkl", "rb") as f:
     X, Y = pickle.load(f)

# X.shape = (100, 64, 32, 1)
# Y.shape = (100,)

# Take a look at the weights

#model_file = "AI_scenario_128x128_3-Conv2d-x32-f4x4_pad-valid_1-bins_1573817953.h5"
model_file = "AI_scenario_32x32_3-Conv2d-x16-f4x4_pad-valid_1-bins_1574324670.h5"
model = load_model(model_file, custom_objects={"tf": tf}, compile=False)

# See attributes of the model:
# dir(model)


############# SALIENCY MAPS ##############################

# Hide warnings on Jupyter Notebook
import warnings
warnings.filterwarnings('ignore')

# I need to swap the last dense layer activation function to a linear one, because
# the model uses a sigmoid function to predict the binary class. (similar to softmax)
model = swap_function_to_linear(model, "output")

# THE RESULTS ARE THE SAME, WETHER IF I CHANGE THE ACTIVATION FUNCTION OR NOT
# THERE's NO IMPACT ON FINAL ACTIVATION FUNCTION DUE TO THE FACT THAT IS A BINARY CLASSIFICATION?

images = X[0], X[1], X[2]
labs = Y[0], Y[1], Y[2]
visualize_images(images)


plot_saliency(model, 'output', images, backprop_mods=[None, 'guided', 'relu'], labels = labs)

# Repeat with grad_modifier = 'negate'.
# This tells us what parts of the image contributes negatively to the output.
# plot_saliency(model, 'output', images, backprop_mods=[None, 'guided', 'relu'], grad_mod= 'negate')


# With other layers. For this model - conv2d, conv2d_1, conv2d_2
layer = 'conv2d_2'
plot_saliency(model, layer, images, backprop_mods=[None, 'guided', 'relu'], labels = labs)