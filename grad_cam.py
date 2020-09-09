import sys

sys.path.insert(0,'/home/pabswfly/keras-vis' )

import numpy as np
from tensorflow import keras
from vis.visualization import visualize_cam, overlay
from vis.utils import utils
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def visualize_images(images, labels=None):
    """Plot a set of pictures given as input. If also label vector is given, this
    function uses them as a tag for each picture"""

    fig = plt.figure(figsize=(8,4))
    grid = ImageGrid(fig, 111, nrows_ncols=(1, len(images)), axes_pad=0.15,
                 share_all=True, cbar_location="right", cbar_mode="single",
                 cbar_size="7%", cbar_pad=0.15)

    # Plot each of the pictures
    for i, im in enumerate(images):

        plot = grid[i].imshow(np.squeeze(im))

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


def plot_gradCAM(model, images_withidx, layer_name='output', backprop_mod = 'guided', labels=None):
    """Function to plot the Class Activation Maps. Inputs:
        - model: CNN model
        - images_withidx: A set of images matrix X with the associated index from the original dataset
        - layer_name: Desired layel for plotting
        - backprop_mod: Modifier for backpropagation. 'guided' generally returns the best and sharpest maps
        - labels: A list of labels y. If given, it is used as title for each of the subfigures plotted"""

    # Separate images_withidx data into image matrices X and image index
    images = [im[0] for im in images_withidx]
    images_idx = [im[1] for im in images_withidx]
    n_im = len(images)

    # Find index in model for the desired layer
    layer = utils.find_layer_idx(model, layer_name)

    fig = plt.figure(figsize=(24, 8))
    grid = ImageGrid(fig, 111, nrows_ncols=(2, n_im), axes_pad=0.15,
                     share_all=True, cbar_location="right", cbar_mode="single",
                     cbar_size="4%", cbar_pad=0.15)

    # For each of the input images
    for i, im in enumerate(images):

        # Calculates the attention gradient using keras-vis library
        grads = visualize_cam(model, layer, filter_indices=0, seed_input=im, backprop_modifier=backprop_mod)

        # Color regularization. Must be in the same scale as input images
        jet_grads = cm.jet(grads) * 255

        # As I get 3 images in jet_grads, I overlay them one by one in order to obtain superposed map.
        overlay_img = np.squeeze(im)

        for idx in [0, 1, 2]:
            jet_heatmap = jet_grads[..., idx]
            overlay_img = overlay(overlay_img, jet_heatmap)

        # Plot input picture in the first figure row and the respective CAM beneath.
        plot_im = grid[i].imshow(np.squeeze(im), cmap='Blues')
        plot_cam = grid[i + n_im].imshow(overlay_img, cmap='viridis')

        # Draw labels and image index for easier recognition
        if labels:
            grid[i].set_title(str(images_idx[i]) + ' ' + labels[i])
        else:
            grid[i].set_title(str(images_idx[i]))

    # If no backpropagation modifier is given, the default one is called Vanilla
    if backprop_mod==None :
        backprop_mod= 'Vanilla'

    # Graphical parameters
    grid[-1].cax.colorbar(plot_cam)
    #grid[0].set_yticks([3, 35, 67])
    #grid[0].set_yticklabels(['Neandertal', 'European', 'African'])
    plt.suptitle('grad-CAM map for layer {0} with backprop_modifier: {1}'.format(layer_name, backprop_mod))
    plt.savefig('results/gradcam_{0}_{1}.png'.format(layer_name, backprop_mod))



def average_gradcam(model, images, labels, layer_name='output', backprop_mod = 'guided', grad_mod = 'absolute'):
    """Function to plot the average of Saliency Maps for a given class. Inputs:
            - model: CNN model
            - images_withidx: A set of images matrix X with the associated index from the original dataset
            - layer_name: Desired layel for plotting
            - backprop_mod: Modifier for backpropagation. 'guided' generally returns the best and sharpest maps
            - grad_mod: Gradient modifier. Ex: 'absolute', 'negate'.
            - labels: A list of labels y. If given, it is used as title for each of the subfigures plotted"""

    n_im_AI = labels.count('AI')
    n_im_noAI = labels.count('-')
    print(n_im_AI)
    print(n_im_noAI)

    # Find index in model for the desired layer
    layer = utils.find_layer_idx(model, layer_name)

    # Now switch to a more OO interface to exercise more features.
    fig, axs = plt.subplots(figsize=(10, 5), nrows=1, ncols=2, sharex=True)

    AIdict = {}
    AIdict['AI'] = np.zeros((images.shape[1], images.shape[2]))
    AIdict['-'] = np.zeros((images.shape[1], images.shape[2]))

    # For each of the input images
    for i, (im, lab) in enumerate(zip(images, labels)):

        print(i)

        # Calculates the saliency gradient using keras-vis library.
        grads = visualize_cam(model, layer, filter_indices=0, seed_input=im,
                                   backprop_modifier=backprop_mod, grad_modifier=grad_mod)

        AIdict[lab] = AIdict[lab] + grads



    # If no backpropagation modifier is given, the default one is called Vanilla
    if backprop_mod == None:
        backprop_mod = 'Vanilla'

    # Average calculation
    AIdict['AI'] = AIdict['AI']/n_im_AI
    AIdict['-'] = AIdict['-'] / n_im_noAI

    # Graphical parameters
    ax = axs[0]
    ax.imshow(AIdict['AI'], cmap='viridis')
    ax.set_title('AI (%d ims)' % n_im_AI)
    #ax.set_yticks([3, 35, 67])
    #ax.set_yticklabels(['Neandertal', 'European', 'African'])

    ax = axs[1]
    pic = ax.imshow(AIdict['-'], cmap='viridis')
    ax.set_title('no-AI (%d ims)' % n_im_noAI)

    fig.colorbar(pic, ax=axs, shrink=0.5)
    fig.suptitle('Average of Grad-CAM for layer {0} with backprop_modifier: {1}'.format(layer_name, backprop_mod))
    plt.savefig('results/average_gradcam_{0}_{1}.png'.format(layer_name, backprop_mod))


def label_to_AI(labels):
    """Transforms a list of {0, 1} labels into {-, AI} labels"""

    return ['AI' if lab == 1 else '-' for lab in labels]



def test(model, X, Y):
    """Chunk of code used for testing and debugging. Please ignore."""

    images = X[0], X[1], X[2], X[45], X[89]
    labs = Y[0], Y[1], Y[2], Y[45], Y[89]

    plot_gradCAM(model, layer_name='output', images=images, labels=labs)