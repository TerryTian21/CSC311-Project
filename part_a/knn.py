from sklearn.impute import KNNImputer 
from utils import *
import numpy as np
import matplotlib.pyplot as plt


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    matrix = np.array(matrix).T

    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(matrix).T
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc

def plot_accuracy(k_list, item_acc, title):
    plt.figure(figsize=(10,6))
    plt.bar(k_list, item_acc, color="blue", edgecolor="black")
    plt.ylim(bottom=0.5)
    plt.title(title)
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.savefig("../Static/item_filtering.png", dpi=400)
    plt.show()

def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################

    # Initialize Variables
    k_list = [1, 6, 11, 16, 21 , 26]
    best_acc_user, best_k_user = 0, 1
    best_acc_item, best_k_item = 0, 1

    user_acc, item_acc = [], []

    # Lists used for plotting

    for k in k_list:
        curr_acc_user = knn_impute_by_user(sparse_matrix, val_data, k)
        curr_acc_item = knn_impute_by_item(sparse_matrix, val_data, k)
        item_acc.append(curr_acc_item)
        user_acc.append(curr_acc_user)

        # Check acc for user based filtering
        if curr_acc_user > best_acc_user:
            best_k_user, best_acc_user = k, curr_acc_user

        # Check acc for 
        if curr_acc_item > best_acc_item:
            best_k_item, best_acc_item = k, curr_acc_item

    # Obtain Test Accuracies
    test_acc_user = knn_impute_by_user(sparse_matrix, test_data, best_k_user)
    test_acc_item = knn_impute_by_item(sparse_matrix, test_data, best_k_item)

    print(f"The best test acc and K value for user based filter is {test_acc_user*100:.2f}% and k={best_k_user}")
    print(f"The best test acc and K value for item based filter is {test_acc_item*100:.2f}% and k={best_k_item}")

    # plot_accuracy(k_list, user_acc, "Accuracy of User Based Collaborative Filtering")
    plot_accuracy(k_list, item_acc, "Accuracy of Item Based Collaborative Filtering")

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
