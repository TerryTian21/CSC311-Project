from utils import *
from scipy.linalg import sqrtm
import itertools
from tqdm.auto import tqdm
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np


def svd_reconstruct(matrix, k):
    """ Given the matrix, perform singular value decomposition
    to reconstruct the matrix.

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D matrix
    """
    # First, you need to fill in the missing values (NaN) to perform SVD.
    # Fill in the missing values using the average on the current item.
    # Note that there are many options to do fill in the
    # missing values (e.g. fill with 0).
    new_matrix = matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    item_means = np.mean(masked_matrix, axis=0)
    new_matrix = masked_matrix.filled(item_means)

    # Next, compute the average and subtract it.
    item_means = np.mean(new_matrix, axis=0)
    mu = np.tile(item_means, (new_matrix.shape[0], 1))
    new_matrix = new_matrix - mu

    # Perform SVD.
    Q, s, Ut = np.linalg.svd(new_matrix, full_matrices=False)
    s = np.diag(s)

    # Choose top k eigenvalues.
    s = s[0:k, 0:k]
    Q = Q[:, 0:k]
    Ut = Ut[0:k, :]
    s_root = sqrtm(s)

    # Reconstruct the matrix.
    reconst_matrix = np.dot(np.dot(Q, s_root), np.dot(s_root, Ut))
    reconst_matrix = reconst_matrix + mu
    return np.array(reconst_matrix)


def squared_error_loss(data, u, z):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i]
                 - np.sum(u[data["user_id"][i]] * z[q])) ** 2.
    return 0.5 * loss


def update_u_z(train_data, lr, u, z):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    i = \
        np.random.choice(len(train_data["question_id"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]

    # Calculate Error
    error = c - np.dot(u[n], z[q].T)

    # Graident Updates (note that the double negatives cancle out so its +=)
    u[n] += lr * error * z[q]
    z[q] += lr * error * u[n]
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def als(train_data, k, lr, num_iteration, val_data = None):
    """ Performs ALS algorithm, here we use the iterative solution - SGD 
    rather than the direct solution.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))
    
    losses = {
        "train": [],
        "val": []
    }

    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    for _ in range(num_iteration):
        train_loss = squared_error_loss(train_data, u, z)
        if val_data:
            val_loss = squared_error_loss(val_data, u, z)
            losses["val"].append(val_loss)
        u, z = update_u_z(train_data, lr, u, z)
        losses["train"].append(train_loss)

    mat = u @ z.T
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat, losses

def accuracy_fn(preds, truth):
    """
    Gives the prediction accuracy of two lists.

    :param preds: list
    :param truth: list

    """
    return sum([1 if i == j else 0 for i, j in zip(preds, truth)])/len(preds)

def plot_loss(train_data, val_data, k, lr, iter):
    
    _, loss = als(train_data, k, lr, iter, val_data)

    plt.figure(figsize=(12, 9))
    plt.subplots_adjust(hspace=0.5)

    plt.subplot(2,1,1)
    plt.plot(loss["train"], label="train loss")
    plt.xlabel("iteration")
    plt.ylabel("squared error loss")
    plt.title("Train Loss over Time")

    plt.subplot(2,1,2)
    plt.plot(loss["val"], label="val loss")
    plt.xlabel("iteration")
    plt.ylabel("squared error loss")
    plt.title("Val Loss over Time")

    plt.legend()
    plt.savefig("../Static/als_loss.png", dpi=400)
    plt.show()

def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # (SVD) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    num_latent = [3, 5, 7, 9, 11, 15, 20]
    for k in num_latent:
        m = svd_reconstruct(train_matrix, k)
        preds = sparse_matrix_predictions(val_data, m)
        accuracy = accuracy_fn(preds, val_data["is_correct"])
        print(f"Accuracy for k = {k} is {accuracy:4f}")

    #  TODO: print validation and test acc
    m  = svd_reconstruct(train_matrix, 9)
    preds = sparse_matrix_predictions(test_data, m)
    print(accuracy_fn(test_data["is_correct"], preds))
    
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    num_latent = [3, 5, 9, 15, 20]
    lrs = [0.001, 0.01, 0.03, 0.1, 0.3]
    iterations = [10, 20, 50, 100]

    results = {
        "k": [],
        "lr": [],
        "iter": [],
        "acc": []
    }

    for k, lr, iter in tqdm(list(itertools.product(num_latent, lrs, iterations))):

        m, _ = als(train_data, k, lr, iter)
        preds = sparse_matrix_predictions(val_data, m)
        accuracy = accuracy_fn(preds, val_data["is_correct"])
        tqdm.write(f"Accuracy for k = {k} | lr = {lr} | iter = {iter} | acc = {accuracy}")
        results['k'].append(k)
        results['lr'].append(lr)
        results["iter"].append(iter)
        results['acc'].append(accuracy)

    df = pd.DataFrame(results)
    df = df.sort_values(by="acc", ascending=False)
    print(df)

    m , _ = als(train_data, 3, 0.3, 100)
    preds = sparse_matrix_predictions(test_data, m)
    print(accuracy_fn(test_data["is_correct"], preds))
    
    #  TODO: plot data

    plot_loss(train_data, val_data, 3, 0.3, 100)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    np.random.seed(42)
    main()
