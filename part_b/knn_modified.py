from sklearn.impute import KNNImputer
from sklearn.metrics.pairwise import cosine_similarity
from utils import *
import numpy as np
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


def knn_impute_by_user(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
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


def normalize_array(array):
    min_val = np.nanmin(array)
    max_val = np.nanmax(array)
    return (array - min_val) / (max_val - min_val)


def scale_array(array, scale_factor):
    return array * scale_factor


def append_data_as_new_question(matrix, data):
    # Append data as new question
    num_users, num_questions = matrix.shape
    new_matrix = np.zeros((num_users, num_questions + 1))
    new_matrix[:, :-1] = matrix
    new_matrix[:, -1] = data
    return new_matrix


def get_year_array(student_meta_data):
    user_dob = np.array(
        [
            student_meta_data["date_of_birth"][i]
            for i in range(len(student_meta_data["user_id"]))
        ]
    )

    # print("User dob: ", user_dob)
    # Extract the year from each date
    user_year = np.array(
        [
            dob.year if dob is not None and dob.year <= 2024 else np.nan
            for dob in user_dob
        ]
    )

    # print("User year: ", user_year)
    return user_year


def get_month_array(student_meta_data):
    user_dob = np.array(
        [
            student_meta_data["date_of_birth"][i]
            for i in range(len(student_meta_data["user_id"]))
        ]
    )

    # Extract the month from each date
    user_month = np.array(
        [
            dob.month if dob is not None and dob.year <= 2024 else np.nan
            for dob in user_dob
        ]
    )
    # print("User month: ", user_month)
    return user_month


def get_premium_array(student_meta_data):
    user_premium = np.array(
        [
            (
                student_meta_data["premium_pupil"][i]
                if student_meta_data["premium_pupil"][i] is not None
                else np.nan
            )
            for i in range(len(student_meta_data["user_id"]))
        ]
    )
    return user_premium

def get_gender_array(student_meta_data):
    gender_array = np.array(
        [
            student_meta_data["gender"][i] if student_meta_data["gender"][i] is not None else np.nan
            for i in range(len(student_meta_data["user_id"]))
        ]
    )
    return gender_array

def eval_adding_premium(sparse_matrix, student_meta_data, val_data):
    new_sparse_matrix = append_data_as_new_question(sparse_matrix, get_premium_array(student_meta_data))

    valid_acc = []
    for k in [11]:
        print("\nk == ", k)
        valid_acc.append(knn_impute_by_user(new_sparse_matrix, val_data, k))

    scale_factors = [0.1, 0.9]
    for scale in scale_factors:
        print("\nPremium scale == ", scale)
        new_sparse_matrix = append_data_as_new_question(sparse_matrix, get_premium_array(student_meta_data) * scale)
        valid_acc.append(knn_impute_by_user(new_sparse_matrix, val_data, 11))

    print(valid_acc)
    print("Max accuracy: ", max(valid_acc))
    return append_data_as_new_question(sparse_matrix, get_premium_array(student_meta_data) * scale_factors[np.argmax(valid_acc) - 1])

def eval_adding_gender(sparse_matrix, student_meta_data, val_data):
    new_sparse_matrix = append_data_as_new_question(sparse_matrix, get_gender_array(student_meta_data))

    valid_acc = []
    for k in [11]:
        print("\nk == ", k)
        valid_acc.append(knn_impute_by_user(new_sparse_matrix, val_data, k))

    scale_factors = [0.1, 0.9]
    for scale in scale_factors:
        print("\nPremium scale == ", scale)
        new_sparse_matrix = append_data_as_new_question(sparse_matrix, get_premium_array(student_meta_data) * scale)
        valid_acc.append(knn_impute_by_user(new_sparse_matrix, val_data, 11))

    print(valid_acc)
    print("Max accuracy: ", max(valid_acc))
    return append_data_as_new_question(sparse_matrix, get_premium_array(student_meta_data) * scale_factors[np.argmax(valid_acc) - 1])

def eval_adding_gender(sparse_matrix, student_meta_data, val_data):
    new_sparse_matrix = append_data_as_new_question(sparse_matrix, get_gender_array(student_meta_data))

    valid_acc = []
    for k in [11]:
        print("\nk == ", k)
        valid_acc.append(knn_impute_by_user(new_sparse_matrix, val_data, k))

    scale_factors = [0.1, 0.5]
    for i, scale in enumerate(scale_factors):
        print("\nGender scale == ", scale)
        new_sparse_matrix = append_data_as_new_question(sparse_matrix, get_gender_array(student_meta_data) * scale)
        valid_acc.append(knn_impute_by_user(new_sparse_matrix, val_data, 11))

    print(valid_acc)
    print("Max accuracy: ", max(valid_acc), "at scale factor: ", scale_factors[np.argmax(valid_acc)-1])
    return append_data_as_new_question(sparse_matrix, get_gender_array(student_meta_data) * scale_factors[np.argmax(valid_acc)-1])


def plot_accuracy(scale_factors, valid_acc, title, filename):
    plt.figure(figsize=(10, 6))
    plt.bar(
        [str(sf) for sf in scale_factors],
        valid_acc,
        width=0.2,
        color="blue",
        edgecolor="black",
    )  # Adjust the width parameter to make bars smaller
    plt.ylim(
        [min(valid_acc) - 0.01, max(valid_acc) + 0.01]
    )  # Adjust the y-axis range to better visualize differences
    plt.title(title)
    plt.xlabel("Scale Factors")
    plt.ylabel("Accuracy")
    plt.savefig(f"../Static/{filename}.png", dpi=400)
    plt.show()


def test_year(year_array, sparse_matrix, val_data):
    normalized_year_array = normalize_array(year_array)
    # month_array = get_month_array(student_meta_data)

    k_values = [6, 11, 16, 21, 26]
    # Not normalized with year
    sparse_matrix = append_data_as_new_question(sparse_matrix, year_array)

    valid_acc = []
    for k in k_values:
        valid_acc.append(knn_impute_by_user(sparse_matrix, val_data, k))

    print("Not Normalized")
    for i in range(len(valid_acc)):
        print(f"Validation Accuracy for k = {k_values[i]} is {valid_acc[i]:.4f}")

    # Normalized with year
    sparse_matrix = append_data_as_new_question(sparse_matrix, normalized_year_array)
    valid_acc = []
    for k in k_values:
        valid_acc.append(knn_impute_by_user(sparse_matrix, val_data, k))

    print("Normalized")
    for i in range(len(valid_acc)):
        print(f"Validation Accuracy for k = {k_values[i]} is {valid_acc[i]:.4f}")

    # Scaled with year
    scale_factors = [0.1, 0.5, 1, 2, 10]
    valid_acc = []
    for scale_factor in scale_factors:
        sparse_matrix = append_data_as_new_question(
            sparse_matrix, scale_array(normalized_year_array, scale_factor)
        )
        valid_acc.append(knn_impute_by_user(sparse_matrix, val_data, 11))

    # print("Scaled (with normalized year)")
    # for i, scale_factor in enumerate(scale_factors):
    #     print(f"Scaled by {scale_factor}")
    #     print(f"Validation Accuracy for k = {11} is {valid_acc[i]:.4f}")

    plot_accuracy(
        scale_factors,
        valid_acc,
        "Accuracy of User-based Filtering With Year Column Being Scaled With Different Factors",
        "knn_modified_year_filtering",
    )


def test_month(month_array, sparse_matrix, val_data):
    normalized_month_array = normalize_array(month_array)
    # Test normalized month with different scale factors for k = 11
    scale_factors = [0.1, 0.5, 1, 2, 10]
    valid_acc = []
    for scale_factor in scale_factors:
        sparse_matrix = append_data_as_new_question(
            sparse_matrix, scale_array(normalized_month_array, scale_factor)
        )
        valid_acc.append(knn_impute_by_user(sparse_matrix, val_data, 11))

    print("Scaled (with normalized month)")
    for i, scale_factor in enumerate(scale_factors):
        print(f"Scaled by {scale_factor}")
        print(f"Validation Accuracy for k = {11} is {valid_acc[i]:.4f}")

    plot_accuracy(
        scale_factors,
        valid_acc,
        "Accuracy of User-based Filtering With Month Column Being Scaled With Different Factors",
        "knn_modified_month_filtering",
    )


def main():
    # Get the root of the project
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Update the paths to the data files
    sparse_matrix = load_train_sparse(os.path.join(project_root, "data")).toarray()
    val_data = load_valid_csv(os.path.join(project_root, "data"))
    test_data = load_public_test_csv(os.path.join(project_root, "data"))
    student_meta_data = load_student_csv(os.path.join(project_root, "data"))
    year_array = get_year_array(student_meta_data)

    sparse_matrix = append_data_as_new_question(sparse_matrix, year_array)
    test_year(year_array, sparse_matrix, val_data)

    month_array = get_month_array(student_meta_data)
    sparse_matrix = append_data_as_new_question(sparse_matrix, month_array)
    test_month(month_array, sparse_matrix, val_data)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
