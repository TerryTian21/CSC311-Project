from sklearn.impute import KNNImputer
from sklearn.metrics.pairwise import cosine_similarity
from utils import *
import numpy as np
import os
import matplotlib.pyplot as plt


def calculate_dob_similarity(user_dob):
    # Compute similarity matrix based on age similarity
    num_users = len(user_dob)
    age_similarity_matrix = np.zeros((num_users, num_users))
    for i in range(num_users):
        for j in range(num_users):
            # Calculate age similarity (e.g., inverse of absolute age difference)
            age_similarity_matrix[i, j] = 1 / (1 + abs(user_dob[i] - user_dob[j]))
    return age_similarity_matrix


def knn_impute_with_dob(matrix, user_age, valid_data, k, age_weight=0.5):
    num_users, num_questions = matrix.shape
    imputed_matrix = np.copy(matrix)

    # Calculate age similarity matrix
    age_similarity_matrix = calculate_dob_similarity(user_age)

    for i in range(num_users):
        for j in range(num_questions):
            if np.isnan(matrix[i][j]):
                # Compute combined similarity using age similarity and traditional KNN similarity
                combined_similarity = (
                    age_weight * age_similarity_matrix[i, :]
                    + (1 - age_weight)
                    * cosine_similarity(matrix[i, :].reshape(1, -1), matrix)[0]
                )

                # Find k-nearest neighbors based on combined similarity
                neighbor_indices = np.argsort(combined_similarity)[-k:]

                # Impute missing value using weighted average of neighbors
                imputed_value = np.sum(
                    matrix[neighbor_indices, j] * combined_similarity[neighbor_indices]
                ) / np.sum(combined_similarity[neighbor_indices])

                # Update imputed value in the matrix
                imputed_matrix[i][j] = imputed_value

    # Evaluate accuracy using validation data
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(imputed_matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def plot_accuracy(k_list, item_acc, title):
    plt.figure(figsize=(10, 6))
    plt.bar(k_list, item_acc, color="blue", edgecolor="black")
    plt.ylim(bottom=0.5)
    plt.title(title)
    plt.xlabel("K")
    plt.ylabel("Accuracy")
    plt.savefig("../Static/item_filtering.png", dpi=400)
    plt.show()


def main():
    # Get the root of the project
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Update the paths to the data files
    sparse_matrix = load_train_sparse(os.path.join(project_root, "data")).toarray()
    val_data = load_valid_csv(os.path.join(project_root, "data"))
    test_data = load_public_test_csv(os.path.join(project_root, "data"))
    student_meta_data = load_student_csv(os.path.join(project_root, "data"))
    print(student_meta_data)

    # print("Sparse matrix:")
    # print(sparse_matrix)
    # print("Shape of sparse matrix:")
    # print(sparse_matrix.shape)
    # print("Student meta data:")

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################

    # Initialize Variables
    k_list = [1, 6, 11, 16, 21, 26]
    best_acc_user, best_k_user = 0, 1

    user_acc = []
    acc = knn_impute_with_dob(
        sparse_matrix, student_meta_data["date_of_birth"], val_data, k_list[0]
    )

    # Lists used for plotting

    # for k in k_list:
    #     curr_acc_user = knn_impute_by_user(sparse_matrix, val_data, k)
    #     user_acc.append(curr_acc_user)

    #     # Check acc for user based filtering
    #     if curr_acc_user > best_acc_user:
    #         best_k_user, best_acc_user = k, curr_acc_user

    # # Obtain Test Accuracies
    # test_acc_user = knn_impute_by_user(sparse_matrix, test_data, best_k_user)

    # print(
    #     f"The best test acc and K value for user based filter is {test_acc_user*100:.2f}% and k={best_k_user}"
    # )

    # plot_accuracy(k_list, user_acc, "Accuracy of User Based Collaborative Filtering")

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
