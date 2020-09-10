import sys

# I installed keras-vis manually because at the time the pip version had some bugs.
sys.path.insert(0,'/home/pabswfly/keras-vis' )

import numpy as np
from vis.visualization import visualize_saliency
from vis.utils import utils
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt


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



def plot_saliency(model, images_withidx, layer_name='output', backprop_mod = 'guided', grad_mod = 'absolute', labels = None):
    """Function to plot the Saliency Maps. Inputs:
        - model: CNN model
        - images_withidx: A set of images matrix X with the associated index from the original dataset
        - layer_name: Desired layel for plotting
        - backprop_mod: Modifier for backpropagation. 'guided' generally returns the best and sharpest maps
        - grad_mod: Gradient modifier. Ex: 'absolute', 'negate'.
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

        # Calculates the saliency gradient using keras-vis library.
        grads = visualize_saliency(model, layer, filter_indices=0, seed_input=im,
                                   backprop_modifier=backprop_mod, grad_modifier= grad_mod)

        # Plot input picture in the first row and the respective saliency map below.
        plot_im = grid[i].imshow(np.squeeze(im), cmap='Blues')
        plot_sal = grid[i+n_im].imshow(grads, cmap='hot')

        # Draw labels and image index for easier recognition
        if labels:
            grid[i].set_title(str(images_idx[i]) + ' ' + labels[i])
        else:
            grid[i].set_title(str(images_idx[i]))


    # If no backpropagation modifier is given, the default one is called Vanilla
    if backprop_mod == None:
        backprop_mod = 'Vanilla'

    # Graphical parameters
    grid[-1].cax.colorbar(plot_sal)
    #grid[0].set_yticks([3, 35, 67])
    #grid[0].set_yticklabels(['Neandertal', 'European', 'African'])
    plt.suptitle('Saliency map for layer {0} with backprop_modifier: {1}'.format(layer_name, backprop_mod))
    plt.savefig('results/saliency_{0}_{1}.png'.format(layer_name, backprop_mod))



def average_saliency(model, images, labels, layer_name='output', backprop_mod = 'guided', grad_mod = 'absolute'):
    """Function to plot the average of Saliency Maps for a given class. Inputs:
            - model: CNN model
            - images_withidx: A set of images matrix X with the associated index from the original dataset
            - layer_name: Desired layel for plotting
            - backprop_mod: Modifier for backpropagation. 'guided' generally returns the best and sharpest maps
            - grad_mod: Gradient modifier. Ex: 'absolute', 'negate'.
            - labels: A list of labels y. If given, it is used as title for each of the subfigures plotted"""

    # Count number of pictures labelled as AI and no-AI (-).
    n_im_AI = labels.count('AI')
    n_im_noAI = labels.count('-')
    print(f'Number of images with AI: {n_im_AI}')
    print(f'Number of images with no AI: {n_im_noAI}')

    # Find index in model for the desired layer
    layer = utils.find_layer_idx(model, layer_name)

    # Initialize empty variables
    AIdict = {}
    AIdict['AI'] = np.zeros((images.shape[1], images.shape[2]))
    AIdict['-'] = np.zeros((images.shape[1], images.shape[2]))

    # For each of the input images
    for i, (im, lab) in enumerate(zip(images, labels)):

        print(f'processing image {i}')

        # Calculates the saliency gradient using keras-vis library.
        grads = visualize_saliency(model, layer, filter_indices=0, seed_input=im,
                                   backprop_modifier=backprop_mod, grad_modifier=grad_mod)

        # Add it to the respective labelled class
        AIdict[lab] = AIdict[lab] + grads



    # If no backpropagation modifier is given, the default one is called Vanilla
    if backprop_mod == None:
        backprop_mod = 'Vanilla'

    # Average calculation for each class
    AIdict['AI'] = AIdict['AI']/n_im_AI
    AIdict['-'] = AIdict['-'] / n_im_noAI

    # Difference between average saliency for each class
    diff = AIdict['AI'] - AIdict['-']

    # Plotting the difference in saliency map
    fig, ax = plt.subplots(figsize=(10, 7))
    pic = ax.imshow(diff, cmap='hot', vmin=-0.1, vmax=0.1)
    #ax.set_yticks([3, 35, 67])
    #ax.set_yticklabels(['Neandertal', 'European', 'African'])
    fig.colorbar(pic, ax=ax)
    fig.suptitle('Difference of average saliency maps for AI and no-AI class')
    plt.savefig('results/difference_AI.png'.format(layer_name, backprop_mod))
    plt.clf()

    # Plotting the two average saliency maps for each class respectively
    fig, axs = plt.subplots(figsize=(12, 7), nrows=1, ncols=2, sharex=True)

    ax = axs[0]
    ax.imshow(AIdict['AI'], cmap='hot', vmin=0, vmax=0.2)
    ax.set_title('AI (%d ims)' % n_im_AI)
    #ax.set_yticks([3, 35, 67])
    #ax.set_yticklabels(['Neandertal', 'European', 'African'])

    ax = axs[1]
    pic = ax.imshow(AIdict['-'], cmap='hot', vmin=0, vmax=0.2)
    ax.set_title('no-AI (%d ims)' % n_im_noAI)

    fig.colorbar(pic, ax=axs, shrink=0.5)
    fig.suptitle('Average of Saliency maps for layer {0} with backprop_modifier: {1}'.format(layer_name, backprop_mod))
    plt.savefig('results/average_saliency_{0}_{1}.png'.format(layer_name, backprop_mod))



def test(model, X, Y):
    """Chunk of code used for testing and debugging. Please ignore."""

    images = X[0], X[1], X[2]
    labs = Y[0], Y[1], Y[2]

    # Visualize the effect of different backpropagation modifiers
    plot_saliency(model, 'output', images, backprop_mods=[None, 'guided', 'relu'], labels = labs)

    # With grad_mod='negate', it tells us what parts of the image contributes negatively to the output.
    plot_saliency(model, 'output', images, backprop_mods=[None, 'guided', 'relu'], grad_mod= 'negate')

    # With other convolutional layers. For this model - conv2d, conv2d_1, conv2d_2
    layer = 'conv2d_2'
    plot_saliency(model, layer, images, backprop_mods='guided', labels = labs)