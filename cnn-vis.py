import os
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore')

    import tensorflow as tf
    from tensorflow.keras.models import load_model
    import pickle
    import argparse

    from saliency import plot_saliency
    from grad_cam import plot_gradCAM
    from activation_max import plot_weights, plot_actmax, swap_function_to_linear


def label_to_AI(labels):
    """Transforms a list of {0, 1} labels into {-, AI} labels"""

    return ['AI' if lab == 1 else '-' for lab in labels]



if __name__ == "__main__":

    # Parser object to collect user input from terminal
    parser = argparse.ArgumentParser(description='Application of keras-vis library to visualize Saliency Maps')
    parser.add_argument('-f', help='Type of map to visualize or utility function',
                        choices=['saliency', 'grad-cam', 'activation-max', 'filter-weights', 'model-summary', 'get-labels'],
                        required=True, type=str)
    parser.add_argument('-m', help='.pkl file with the Convolutional Neural Network model to be analized',
                        required=True, type=str)
    parser.add_argument('-d', help='.pkl file with labelled images that were used to train the CNN',
                        required=True, type=str)
    parser.add_argument('-im', help='Selected images to use for saliency maps or grad-CAM maps', required=True, nargs='+', type=int)
    parser.add_argument('-l', help='Convolutional layer from model to plot', required=False, type=str)

    # Get argument values from parser
    args = parser.parse_args()
    model_file = args.m
    data_file = args.d
    indeces = args.im
    layer_name = args.l
    function = args.f

    # loading the data. (1 == adaptive introgression, 0 otherwise)
    # X.shape = (100, 64, 32, 1)
    # Y.shape = (100,)
    with open(data_file, "rb") as f:
         X, Y = pickle.load(f)


    # loading the model
    model = load_model(model_file, custom_objects={"tf": tf}, compile=False)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # I need to swap the last dense layer activation function to a linear one, because
    # the model uses a sigmoid function to predict the binary class. (similar to softmax)
    # Softmax activation function gives a suboptimal result in keras-vis documentation.
    model = swap_function_to_linear(model, "output")

    # In order to check model classification performance:
    #loss, acc = model.evaluate(X, Y, verbose=2)
    #print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

    # See attributes of the model:
    # dir(model)


    # Separate data in matrix X with picture index, and labels vector y with y={0 --> -, 1 --> AI}
    images_withidx = [(X[i], i) for i in indeces]
    labs = [Y[i] for i in indeces]
    labs2 = label_to_AI(labs)

    # Check if folder exists to store output .png files, else create it
    if not os.path.exists('results'):
        os.makedirs('results')


# ---------------------------------------------------------------------------------------------------------------------

    # Saliency maps
    if function == 'saliency':
        for b_m in [None, 'guided', 'relu']:
            plot_saliency(model, images_withidx, layer_name=layer_name, labels = labs2, backprop_mod=b_m)

    # Gradient-Class Activation Maps
    elif function == 'grad-cam':
        for b_m in [None, 'guided', 'relu']:
            plot_gradCAM(model, images_withidx, layer_name=layer_name, labels = labs2, backprop_mod=b_m)

    # Activation Maximization Maps
    elif function == 'activation-max':
        plot_actmax(model, layer_name)

    # Plot weights from each filter of a given convolutional layer
    elif function == 'filter-weights':
        plot_weights(model, layer_name)

    # Print model summary
    elif function =='model-summary':
        print(model.summary())

    # Print input picture labels
    elif function == 'get-labels':
        for i, lab in enumerate(Y):
            print('(' + str(i) + '): ' + str(lab) + ' ')



# Example of terminal command to run the code:
# python3 cnn-vis.py -f grad-cam -m data/AI_scenario_32x32_3-Conv2d-x16-f4x4_pad-valid_1-bins_1574324670.h5 -d data/test_data_32x32.pkl -l output -im 40 41 42 43 44 45
