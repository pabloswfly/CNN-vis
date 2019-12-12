import sys

sys.path.insert(0,'/home/pabswfly/downloads/keras-vis' )

import numpy as np
import matplotlib.pyplot as plt
from vis.utils import utils
import matplotlib.cm as cm
from vis.visualization import visualize_cam, overlay
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


def plot_gradCAM(model, layer_name, images, labels=None):

    layer = utils.find_layer_idx(model, layer_name)

    fig = plt.figure(figsize=(8,3))
    plt.suptitle('grad-CAM map')

    grid = ImageGrid(fig, 111, nrows_ncols=(1, len(images)), axes_pad=0.15,
                     share_all=True, cbar_location="right", cbar_mode="single",
                     cbar_size="7%", cbar_pad=0.15)

    for i, im in enumerate(images):

        grads = visualize_cam(model, layer, filter_indices=0, seed_input=im)

        # Get better color map with RGB color
        jet_grads = cm.jet(grads) * 255

        # As I get 3 images in jet_grads, we overlay them one by one into the images
        overlay_img = np.squeeze(im)

        for idx in [0, 1, 2]:

            jet_heatmap = jet_grads[..., idx]
            overlay_img = overlay(overlay_img, jet_heatmap)

        plot = grid[i].imshow(overlay_img)

        if labels:
            grid[i].set_title(labels[i])

    grid[-1].cax.colorbar(plot)
    grid[0].set_yticks([3, 35, 67])
    grid[0].set_yticklabels(['Neandertal', 'European', 'African'])
    plt.show()



def label_to_AI(labels):

    ai_labels = ['AI' if lab == 1 else '-' for lab in labels]

    return ai_labels


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


############# CLASS ACTIVATION MAPS ##############################

# Hide warnings on Jupyter Notebook
import warnings
warnings.filterwarnings('ignore')

# I need to swap the last dense layer activation function to a linear one, because
# the model uses a sigmoid function to predict the binary class. (similar to softmax)
model = swap_function_to_linear(model, "output")

images = X[0], X[1], X[2], X[45], X[89]
labs = Y[0], Y[1], Y[2], Y[45], Y[89]

labels = label_to_AI(labs)

visualize_images(images, labels=labels)

plot_gradCAM(model, layer_name='output', images=images, labels=labels)