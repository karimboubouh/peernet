# -------------------
# organize imports
# -------------------
import numpy as np
from algorithms.lr_utils import load_dataset

# from keras.preprocessing import image

# ------------------------
# tunable parameters
# ------------------------
image_size = (64, 64)
num_train_images = 1500
num_test_images = 100
num_channels = 3
epochs = 2000
lr = 0.01

# train_x dimension = {(64*64*3), 1500}
# train_y dimension = {1, 1500}
# test_x dimension  = {(64*64*3), 100}
# test_y dimension  = {1, 100}


# ----------------
# load dataset
# ----------------
train_x, train_y, test_x, test_y, classes = load_dataset()
train_x_flatten = train_x.reshape(train_x.shape[0], -1).T
test_x_flatten = test_x.reshape(test_x.shape[0], -1).T

# ------------------
# standardization
# ------------------
train_x = train_x_flatten / 255.
test_x = test_x_flatten / 255.

print("train_labels : " + str(classes))
print("train_x shape: " + str(train_x.shape))
print("train_y shape: " + str(train_y.shape))
print("test_x shape : " + str(test_x.shape))
print("test_y shape : " + str(test_y.shape))


# ----------------
# define sigmoid
# ----------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ---------------------------
# parameter initialization
# ---------------------------
def init_params(dimension):
    w = np.zeros((dimension, 1))
    b = 0
    return w, b


# ------------------------------
# forward and back propagation
# ------------------------------
def propagate(w, b, X, Y):
    # num of training samples
    m = X.shape[1]

    # forward pass
    A = sigmoid(np.dot(w.T, X) + b)
    cost = (-1 / m) * (np.sum(np.multiply(Y, np.log(A)) + np.multiply((1 - Y), np.log(1 - A))))

    # back propagation
    dw = (1 / m) * (np.dot(X, (A - Y).T))
    db = (1 / m) * (np.sum(A - Y))

    cost = np.squeeze(cost)

    # gradient dictionary
    grads = {"dw": dw, "db": db}

    return grads, cost


# ------------------
# gradient descent
# ------------------
def optimize(w, b, X, Y, epochs, lr):
    costs = []
    for i in range(epochs):
        # calculate gradients
        grads, cost = propagate(w, b, X, Y)

        # get gradients
        dw = grads["dw"]
        db = grads["db"]

        # update rule
        w = w - (lr * dw)
        b = b - (lr * db)

        if i % 100 == 0:
            costs.append(cost)
            print("cost after %i epochs: %f" % (i, cost))

    # param dict
    params = {"w": w, "b": b}

    # gradient dict
    grads = {"dw": dw, "db": db}

    return params, grads, costs


# -------------------------------------
# make prediction on multiple images
# -------------------------------------
def predict(w, b, X):
    m = X.shape[1]
    Y_predict = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        if A[0, i] <= 0.5:
            Y_predict[0, i] = 0
        else:
            Y_predict[0, i] = 1

    return Y_predict


# -------------------------------------
# make prediction on a single image
# -------------------------------------
def predict_image(w, b, X):
    Y_predict = None
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        if A[0, i] <= 0.5:
            Y_predict = 0
        else:
            Y_predict = 1

    return Y_predict


# ----------------------
# logistic regression
# ----------------------
def model(X_train, Y_train, X_test, Y_test, epochs, lr):
    w, b = init_params(X_train.shape[0])
    params, grads, costs = optimize(w, b, X_train, Y_train, epochs, lr)

    w = params["w"]
    b = params["b"]

    Y_predict_train = predict(w, b, X_train)
    Y_predict_test = predict(w, b, X_test)

    print("train_accuracy: {} %".format(100 - np.mean(np.abs(Y_predict_train - Y_train)) * 100))
    print("test_accuracy : {} %".format(100 - np.mean(np.abs(Y_predict_test - Y_test)) * 100))

    lr_model = {"costs": costs,
                "Y_predict_test": Y_predict_test,
                "Y_predict_train": Y_predict_train,
                "w": w,
                "b": b,
                "learning_rate": lr,
                "epochs": epochs}

    return lr_model


# activate the logistic regression model
d = model(train_x, train_y, test_x, test_y, 2000, lr)

index = 5
pic = test_y[0, index]
pred = d['Y_predict_test'][0, index]
print(f"y = {classes[int(pic)].decode('utf-8')}, you predicted that it is {classes[int(pred)].decode('utf-8')}")
# plt.imshow(test_x[:, index].reshape((64, 64, 3)))
# plt.show()
# print("y = " + str(test_y[0, index]) + ", you predicted that it is a \"" + classes[
#     d["Y_predict_test"][0, index]].decode("utf-8") + "\" picture.")

# ------------------------------
# test images using our model
# ------------------------------
# test_img_paths = [
#     "G:\\workspace\\machine-intelligence\\deep-learning\\logistic-regression\\dataset\\test\\airplane\\image_0723.jpg",
#     "G:\\workspace\\machine-intelligence\\deep-learning\\logistic-regression\\dataset\\test\\airplane\\image_0713.jpg",
#     "G:\\workspace\\machine-intelligence\\deep-learning\\logistic-regression\\dataset\\test\\bike\\image_0782.jpg",
#     "G:\\workspace\\machine-intelligence\\deep-learning\\logistic-regression\\dataset\\test\\bike\\image_0799.jpg",
#     "G:\\workspace\\machine-intelligence\\deep-learning\\logistic-regression\\dataset\\test\\bike\\test_1.jpg"]
#
# for test_img_path in test_img_paths:
#     img_to_show = cv2.imread(test_img_path, -1)
#     img = image.load_img(test_img_path, target_size=image_size)
#     x = image.img_to_array(img)
#     x = x.flatten()
#     x = np.expand_dims(x, axis=1)
#     predict = predict_image(myModel["w"], myModel["b"], x)
#     predict_label = ""
#
#     if predict == 0:
#         predict_label = "airplane"
#     else:
#         predict_label = "bike"
#
#     # display the test image and the predicted label
#     cv2.putText(img_to_show, predict_label, (30, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#     cv2.imshow("test_image", img_to_show)
#     key = cv2.waitKey(0) & 0xFF
#     if (key == 27):
#         cv2.destroyAllWindows()
