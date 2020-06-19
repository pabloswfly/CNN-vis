import os
import warnings
import numpy as np
with warnings.catch_warnings():
    warnings.filterwarnings('ignore')

    import tensorflow as tf
    from tensorflow.keras.models import load_model
    import pickle
    import argparse

    from saliency import plot_saliency, average_saliency
    from grad_cam import plot_gradCAM, average_gradcam
    from activation_max import plot_weights, plot_actmax, swap_function_to_linear


def label_to_AI(labels):
    """Transforms a list of {0, 1} labels into {-, AI} labels"""

    return ['AI' if lab == 1 else '-' for lab in labels]



if __name__ == "__main__":

    # Parser object to collect user input from terminal
    parser = argparse.ArgumentParser(description='Application of keras-vis library to visualize how a CNN works')
    parser.add_argument('function', help='Type of map to visualize or utility function',
                        choices=['saliency', 'average-saliency', 'average-gradcam', 'grad-cam', 'activation-max',
                                 'filter-weights', 'model-summary', 'get-labels'], type=str)
    parser.add_argument('model', help='.pkl file with the Convolutional Neural Network model to be analized', type=str)
    parser.add_argument('-d', '--data', help='.pkl file with labelled images that were used to train the CNN', type=str)
    parser.add_argument('-im', '--images', help='Selected images to use for saliency maps or grad-CAM maps', nargs='+', type=int)
    parser.add_argument('-l', '--layer', help='Convolutional layer from model to plot', default='output', type=str)
    parser.add_argument('-bm', '--backprop', help='Backpropagation modifier', default='guided',
                        choices=[None, 'guided', 'rectified', 'deconv', 'relu'], type=str)

    # Get argument values from parser
    args = parser.parse_args()
    indeces = args.images
    layer_name = args.layer


    # loading the model
    model = load_model(args.model, custom_objects={"tf": tf}, compile=False)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])


    # I need to swap the last dense layer activation function to a linear one, because
    # the model uses a sigmoid function to predict the binary class. (similar to softmax)
    # Softmax activation function gives a suboptimal result in keras-vis documentation.
    model = swap_function_to_linear(model, "output")
    #model = swap_function_to_linear(model, "preds")

    # In order to check model classification performance:
    #loss, acc = model.evaluate(X, Y, verbose=2)
    #print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

    if args.data:
        # loading the data. (1 == adaptive introgression, 0 otherwise)
        # X.shape = (100, 64, 32, 1)
        # Y.shape = (100,)
        with open(args.data, "rb") as file:
            X, Y = pickle.load(file)
            if len(X.shape) == 3:
                X = np.expand_dims(X, axis=-1)
                print(f'X shape is: {X.shape}')

        if args.images:
            # Separate data in matrix X with picture index, and labels vector y with y={0 --> -, 1 --> AI}
            images_withidx = [(X[i], i) for i in indeces]
            labs = [Y[i] for i in indeces]
            labs2 = label_to_AI(labs)

    # See attributes of the model:
    # dir(model)

    # Check if folder exists to store output .png files, else create it
    if not os.path.exists('results'):
        os.makedirs('results')

# ---------------------------------------------------------------------------------------------------------------------

    # Saliency maps
    if args.function == 'saliency' and args.data and args.images and args.layer:
            plot_saliency(model, images_withidx, layer_name=layer_name, backprop_mod=args.backprop)

    elif args.function == 'average-saliency' and args.data and args.layer:
            average_saliency(model, X, labels=label_to_AI(Y), layer_name=layer_name, backprop_mod=args.backprop)

    # Gradient-Class Activation Maps
    elif args.function == 'grad-cam' and args.data and args.images and args.layer:
            plot_gradCAM(model, images_withidx, layer_name=layer_name, backprop_mod=args.backprop)

    elif args.function == 'average-gradcam' and args.data and args.layer:
        average_gradcam(model, X, labels=label_to_AI(Y), layer_name=layer_name, backprop_mod=args.backprop)

    # Activation Maximization Maps
    elif args.function == 'activation-max':
        plot_actmax(model, backprop_mod=args.backprop)

    # Plot weights from each filter of a given convolutional layer
    elif args.function == 'filter-weights' and args.layer:
        plot_weights(model, layer_name)

    # Print model summary
    elif args.function =='model-summary':
        print(model.summary())

    # Print input picture labels
    elif args.function == 'get-labels' and args.data:
        for i, lab in enumerate(Y):
            print('(' + str(i) + '): ' + str(lab) + ' ')



# Example of terminal command to run the code:
# python3 cnn-vis.py average-saliency data/NEW_cnn_1562604956.hdf5 -d data/genmat.pkl -l output -im 0 1 2 5 10 18 19 31 51 72
