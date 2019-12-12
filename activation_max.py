import sys

sys.path.insert(0,'/home/pabswfly/downloads/keras-vis' )

import numpy as np
import matplotlib.pyplot as plt
from vis.utils import utils
import tensorflow as tf
from mpl_toolkits.axes_grid1 import ImageGrid

# Keras is part of tensorflow in versions 2.0 onwards
from tensorflow import keras

from tensorflow.python.keras.models import load_model
import pickle


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

    # If this line returns ValueError: Unknown initializer: GlorotUniform:
    # PROBLEM WITH THIS LINE
    model = utils.apply_modifications(model)

    return model




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



################# ACTIVATION PLOTS ##########################

# imresize from scipy package only works with versions scipy==0.2.*
from vis.visualization import visualize_activation

layer = utils.find_layer_idx(model, 'output')

# Filter indices points to the output node we want to maximize.
# Because here we're working with binary classification, there's only one
img = visualize_activation(model, layer, filter_indices=0,
                           tv_weight=1e-3, lp_norm_weight=0.)
plt.imshow(img[...,0])
plt.title("Without swapping Softmax function")
plt.savefig('Actmax_softmax.png')

model = swap_function_to_linear(model, 'output')


images = X[0], X[1], X[2]
labs = Y[0], Y[1], Y[2]
#visualize_images(images)


# Filter indices points to the output node we want to maximize.
# Because here we're working with binary classification, there's only one
img = visualize_activation(model, layer,
                           tv_weight=1e-3, lp_norm_weight=0.)
plt.imshow(img[...,0])
plt.title("After swapping Softmax function to Linear")
plt.savefig('Actmax_linear.png')


from vis.input_modifiers import Jitter

fig = plt.figure(figsize=(8,3))

grid = ImageGrid(fig, 111, nrows_ncols=(2, 5), axes_pad=0.15,
             share_all=True, cbar_location="right", cbar_mode="single",
             cbar_size="7%", cbar_pad=0.15)

for i, tv_weight in enumerate([1e-5, 1e-4, 5e-3, 1e-3, 5e-2, 1e-2, 5e-1, 1e-1, 1, 10]):

    img = visualize_activation(model, layer, tv_weight=tv_weight, lp_norm_weight=0.,
                               input_modifiers=[Jitter(16)])

    plot = grid[i].imshow(img[..., 0])

    grid[i].set_title('tv_w: {}'.format(tv_weight))

    grid[-1].cax.colorbar(plot)

    plt.title('Activation Maximization layer {}'.format(layer))
    grid[0].set_yticks([3, 35, 67])
    grid[0].set_yticklabels(['Neandertal', 'European', 'African'])

plt.savefig('Actmax_tv_weights.png')



# Plotting the weights of the first layer
W = model.layers[2].get_weights()
W = np.squeeze(W)
W = W.T

for i, filter in enumerate(W):

    plt.subplot(4, 4, i+1)
    plt.imshow(filter)

plt.show()



# Plotting different filters
from vis.visualization import get_num_filters

layer_name = 'conv2d'
layer = utils.find_layer_idx(model, layer_name)

# Visualize all filters in this layer.
filters = np.arange(get_num_filters(model.layers[layer]))

# Generate input image for each filter.
vis_images = []

for idx in filters:
    img = visualize_activation(model, layer, filter_indices=idx)

    # Utility to overlay text on image.
    img = utils.draw_text(img, 'Filter {}'.format(idx))

    vis_images.append(img)

# Generate stitched image palette with 8 cols.
stitched = utils.stitch_images(vis_images, cols=4)
plt.axis('off')
plt.imshow(stitched)
plt.title(layer_name)
plt.show()