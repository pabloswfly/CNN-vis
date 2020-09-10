import sys

# I installed keras-vis manually because at the time the pip version had some bugs.
sys.path.insert(0,'/home/pabswfly/keras-vis' )

import numpy as np
import matplotlib.pyplot as plt
from vis.utils import utils
from mpl_toolkits.axes_grid1 import ImageGrid
from tensorflow import keras
from vis.input_modifiers import Jitter
from vis.visualization import visualize_activation
# If there's an error with imresize, there are two easy solutions:
# - Use imresize from scipy package from versions scipy==0.2.*
# - Use skimage.transform.resize instead.



def visualize_images(images, labels=None):
    """Plot a set of pictures from the input data. If also a label vector is given, this
    function uses them as a tag for each picture"""

    fig = plt.figure(figsize=(8,4))
    grid = ImageGrid(fig, 111, nrows_ncols=(1, len(images)), axes_pad=0.15,
                 share_all=True, cbar_location="right", cbar_mode="single",
                 cbar_size="7%", cbar_pad=0.15)

    # Plot each of the pictures
    for i, im in enumerate(images):
        plot = grid[i].imshow(np.squeeze(im))

        # Add the picture labels if given
        if labels:
            grid[i].set_title(labels[i])

    # Graphical parameters
    grid[-1].cax.colorbar(plot)
    plt.title('Input images')
    grid[0].set_yticks([3, 35, 67])
    grid[0].set_yticklabels(['Neandertal', 'European', 'African'])
    plt.show()


def swap_function_to_linear(model, layer_name):
    """Given a model and a convolutional layer name, swaps the activation function of the layer for a linear one.
    Output: Returns the model with updated linear activation function"""

    # Find layer index in the model and swap for a linear activation function
    layer = utils.find_layer_idx(model, layer_name)
    model.layers[layer].activation = keras.activations.linear

    # This line is necessary to update the model
    model = utils.apply_modifications(model)

    return model


def plot_weights(model, layer_name):
    """Plot the weights from a given model and layer"""

    # Get weights of desired layer from model
    layer = utils.find_layer_idx(model, layer_name)
    W = model.layers[layer].get_weights()
    W = np.squeeze(W)
    W = W.T
    print(len(W))

    # Plot each filter in the layer
    for i, filter in enumerate(W):

        #TODO: Make subplot flexible, not only for 16 filters in a grid of [4, 4]
        plt.subplot(4, 4, i + 1)
        plt.imshow(filter, cmap='Blues')

    plt.show()


def plot_actmax(model, tv_weight=1e-5, backprop_mod=None):
    """Function to plot the Activation Maximization map. Inputs:
        - model: CNN model
        - layer_name: Desired layel for plotting
        - backprop_mod: Modifier for backpropagation. 'guided' generally returns the best and sharpest maps
        - tv_weight: Total variance weight loss. Needs to be tuned to get accurate layer filters."""

    fig = plt.figure(figsize=(16, 8))
    grid = ImageGrid(fig, 111, nrows_ncols=(2, 4), axes_pad=(0.15, 0.30), share_all=True)

    # summarize filter shapes
    for i, layer in enumerate(model.layers):

        # check for convolutional layer
        if 'conv' not in layer.name:
            if 'output' not in layer.name:
                if 'dense' not in layer.name:
                    if 'preds' not in layer.name:
                        continue

        layer_idx = utils.find_layer_idx(model, layer.name)

        # Calculate activation maximization map. Jitter(16) argument results in sharper saliency maps.
        img = visualize_activation(model, layer_idx, filter_indices=0, backprop_modifier=backprop_mod,
                                   tv_weight=tv_weight, lp_norm_weight=0., input_modifiers=[Jitter(16)])

        grid[i].imshow(img[..., 0], cmap='Blues')

        # Graphical parameters
        grid[i].set_title('{}'.format(layer.name))
        #grid[0].set_yticks([3, 35, 67])
        #grid[0].set_yticklabels(['Neandertal', 'European', 'African'])


    # If no backpropagation modifier is given, the default one is called Vanilla
    if backprop_mod == None:
        backprop_mod = 'Vanilla'

    # Graphical parameters
    plt.suptitle('Activation-maximization map with backprop_modifier: {}'.format(backprop_mod))
    plt.savefig('results/actmax_{}.png'.format(backprop_mod))



def test(model, X, Y):
    """Chunk of code used for testing and debugging. Please ignore."""

    layer = utils.find_layer_idx(model, 'conv2d_2')

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

    fig = plt.figure(figsize=(16, 8))

    grid = ImageGrid(fig, 111, nrows_ncols=(2, 5), axes_pad=0.15,
                 share_all=True, cbar_location="right", cbar_mode="single",
                 cbar_size="7%", cbar_pad=0.15)

    for i, tv_weight in enumerate([1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 5e-3, 1e-3, 5e-2, 1e-2]):

        img = visualize_activation(model, layer, filter_indices=0, tv_weight=tv_weight, lp_norm_weight=0.,
                                   input_modifiers=[Jitter(16)])

        plot = grid[i].imshow(img[..., 0])

        grid[i].set_title('tv_w: {}'.format(tv_weight))

        grid[-1].cax.colorbar(plot)

        plt.suptitle('Activation Maximization layer {}'.format(layer))
        grid[0].set_yticks([3, 35, 67])
        grid[0].set_yticklabels(['Neandertal', 'European', 'African'])

    plt.savefig('results/Actmax_tv_weights.png')




    fig = plt.figure(figsize=(16, 12))

    grid = ImageGrid(fig, 111, nrows_ncols=(3, 3), axes_pad=0.15,
                     share_all=True, cbar_location="right", cbar_mode="single",
                     cbar_size="7%", cbar_pad=0.15)

    j = 0

    for modifier in [None, 'guided', 'relu']:
        for tv_weight in [1e-7, 1e-6, 1e-5]:

            img = visualize_activation(model, layer, filter_indices=0, tv_weight=tv_weight, lp_norm_weight=0.,
                                       input_modifiers=[Jitter(16)], backprop_modifier=modifier)

            plot = grid[j].imshow(img[..., 0])

            grid[j].set_title('tv_w: {}'.format(tv_weight))

            grid[-1].cax.colorbar(plot)

            plt.suptitle('Activation Maximization layer {}'.format(layer))
            grid[0].set_yticks([3, 35, 67])
            grid[0].set_yticklabels(['Neandertal', 'European', 'African'])

            j += 1

    plt.savefig('Actmax_tv_weights_backprop.png')
