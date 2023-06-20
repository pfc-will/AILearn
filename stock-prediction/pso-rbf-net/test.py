import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import MSE

import matplotlib.pyplot as plt

from rbflayer import RBFLayer, InitCentersRandom
from kmeans_initializer import InitCentersKMeans
from pso_initializer import InitCentersPSO
from initializer import InitFromFile
import stocks

mainpath = os.path.dirname(__file__)
mainpath = os.path.join(mainpath, 'data')

def load_stock():
    # Reading & Preprocessing data.
    train_data, test_data = stocks.read_data(preprocessing=True, scaling_method='MinMax')

    # Creating time frames.
    x_train, y_train = stocks.construct_time_frames(train_data, frame_size=64)
    x_test, y_test = stocks.construct_time_frames(test_data, frame_size=64)

    return x_train, y_train, x_test, y_test

def load_txt():

    data = np.loadtxt(os.path.join(mainpath, 'data.txt'))
    X = data[:, :-1]  # except last column
    y = data[:, -1]  # last column only
    return X, y

def test_stock(x_train, y_train, x_test, y_test, initializer):

    title = f" test {type(initializer).__name__} "
    print("-"*20 + title + "-"*20)

    # create RBF network as keras sequential model
    model = Sequential()
    rbflayer = RBFLayer(10,
                        initializer=initializer,
                        beta=2.0,
                        input_shape=x_train.shape[1:])
    outputlayer = Dense(1, use_bias=False)

    model.add(rbflayer)
    model.add(outputlayer)

    model.compile(loss='mean_squared_error',
                  optimizer=RMSprop())
    model.summary()

    # fit and predict
    model.fit(x_train, y_train,
              batch_size=64,
              epochs=200,
              verbose=0)

    y_pred = model.predict(x_test)

    # show graph
    # plt.plot(X, y_pred)  # prediction
    # plt.plot(X, y_test)       # response from data
    # plt.plot([-1, 1], [0, 0], color='black')  # zero line
    # plt.xlim([-1, 1])
    stocks.plot_prediction(y_test, y_pred, 'Real Tesla Stock Price', 'Predicted Tesla Stock Price')

    # plot centers
    # centers = rbflayer.get_weights()[0]
    # widths = rbflayer.get_weights()[1]
    # plt.scatter(centers, np.zeros(len(centers)), s=20*widths)

    # plt.show()

    # # calculate and print MSE
    # y_pred = y_pred.squeeze()
    # print(f"MSE: {MSE(y, y_pred):.4f}")

    # # saving to and loading from file
    # filename = f"./output/rbf_{type(initializer).__name__}.h5"
    # print(f"Save model to file {filename} ... ", end="")
    # model.save(filename)
    # print("OK")

    # # print(f"Load model from file {filename} ... ", end="")
    # # newmodel = load_model(filename,
    # #                       custom_objects={'RBFLayer': RBFLayer})
    # # print("OK")

    # # # check if the loaded model works same as the original
    # # y_pred2 = newmodel.predict(X).squeeze()
    # # print("Same responses: ", all(y_pred == y_pred2))
    # # I know that I compared floats, but results should be identical

    # # save, widths & weights separately
    # np.save("centers", centers)
    # np.save("widths", widths)
    # np.save("weights", outputlayer.get_weights()[0])

def test(X, y, initializer):

    title = f" test {type(initializer).__name__} "
    print("-"*20 + title + "-"*20)

    # create RBF network as keras sequential model
    model = Sequential()
    rbflayer = RBFLayer(10,
                        initializer=initializer,
                        beta=2.0,
                        input_shape=(1,))
    outputlayer = Dense(1, use_bias=False)

    model.add(rbflayer)
    model.add(outputlayer)

    model.compile(loss='mean_squared_error',
                  optimizer=RMSprop())

    # fit and predict
    model.fit(X, y,
              batch_size=50,
              epochs=2000,
              verbose=0)

    y_pred = model.predict(X)

    # show graph
    plt.plot(X, y_pred)  # prediction
    plt.plot(X, y)       # response from data
    plt.plot([-1, 1], [0, 0], color='black')  # zero line
    plt.xlim([-1, 1])

    # plot centers
    centers = rbflayer.get_weights()[0]
    widths = rbflayer.get_weights()[1]
    plt.scatter(centers, np.zeros(len(centers)), s=20*widths)

    plt.show()

    # calculate and print MSE
    y_pred = y_pred.squeeze()
    print(f"MSE: {MSE(y, y_pred):.4f}")

    # saving to and loading from file
    filename = f"./output/rbf_{type(initializer).__name__}.h5"
    print(f"Save model to file {filename} ... ", end="")
    model.save(filename)
    print("OK")

    print(f"Load model from file {filename} ... ", end="")
    newmodel = load_model(filename,
                          custom_objects={'RBFLayer': RBFLayer})
    print("OK")

    # check if the loaded model works same as the original
    y_pred2 = newmodel.predict(X).squeeze()
    print("Same responses: ", all(y_pred == y_pred2))
    # I know that I compared floats, but results should be identical

    # save, widths & weights separately
    np.save("centers", centers)
    np.save("widths", widths)
    np.save("weights", outputlayer.get_weights()[0])


def test_init_from_file(X, y):

    print("-"*20 + " test init from file " + "-"*20)

    # load the last model from file
    filename = f"./output/rbf_InitFromFile.h5"
    print(f"Load model from file {filename} ... ", end="")
    model = load_model(filename,
                       custom_objects={'RBFLayer': RBFLayer})
    print("OK")

    res = model.predict(X).squeeze()  # y was (50, ), res (50, 1); why?
    print(f"MSE: {MSE(y, res):.4f}")

    # load the weights of the same model separately
    rbflayer = RBFLayer(10,
                        initializer=InitFromFile("centers.npy"),
                        beta=InitFromFile("widths.npy"),
                        input_shape=(1,))
    print("rbf layer created")
    outputlayer = Dense(1,
                        kernel_initializer=InitFromFile("weights.npy"),
                        use_bias=False)
    print("output layer created")

    model2 = Sequential()
    model2.add(rbflayer)
    model2.add(outputlayer)

    res2 = model2.predict(X).squeeze()
    print(f"MSE: {MSE(y, res2):.4f}")
    print("Same responses: ", all(res == res2))


if __name__ == "__main__":

    x_train, y_train, x_test, y_test = load_stock()
    # test_stock(x_train, y_train, x_test, y_test, InitCentersPSO(x_train))
    test_stock(x_train, y_train, x_test, y_test, InitCentersPSO(x_train, y_train))

    # X, y = load_txt()

    # test simple RBF Network with random  setup of centers
    # test(X, y, InitCentersRandom(X))

    # # test simple RBF Network with centers set up by k-means
    # test(X, y, InitCentersKMeans(X))
