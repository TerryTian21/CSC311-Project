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
        [min(valid_acc) - 0.005, max(valid_acc) + 0.005]
    )  # Adjust the y-axis range to better visualize differences
    plt.title(title)
    plt.xlabel("Scale Factors")
    plt.ylabel("Accuracy")
    plt.savefig(f"../Static/{filename}.png", dpi=400)
    plt.show()


def plot_pred_year(year_array, sparse_matrix, val_data):
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
    scale_factors = [0.0, 0.1, 0.5, 1, 2, 10]
    valid_acc = []
    for scale_factor in scale_factors:
        new_sparse_matrix = append_data_as_new_question(
            sparse_matrix, scale_array(normalized_year_array, scale_factor)
        )
        valid_acc.append(knn_impute_by_user(new_sparse_matrix, val_data, 11))

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


def plot_pred_month(month_array, sparse_matrix, val_data):
    normalized_month_array = normalize_array(month_array)
    # Test normalized month with different scale factors for k = 11
    scale_factors = [0.0, 0.1, 0.5, 1, 2, 10]
    valid_acc = []
    for scale_factor in scale_factors:
        # Create a copy of the sparse matrix
        new_sparse_matrix = append_data_as_new_question(
            sparse_matrix, scale_array(normalized_month_array, scale_factor)
        )
        valid_acc.append(knn_impute_by_user(new_sparse_matrix, val_data, 11))

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

def plot_pred_gender(sparse_matrix, student_meta_data, val_data):
    normalized_gender_array = normalize_array(get_gender_array(student_meta_data))
    scale_factors = [0.0, 0.1, 0.5, 1, 2, 10]
    valid_acc = []
    for scale_factor in scale_factors:
        new_sparse_matrix = append_data_as_new_question(
            sparse_matrix, scale_array(normalized_gender_array, scale_factor)
        )
        valid_acc.append(knn_impute_by_user(new_sparse_matrix, val_data, 11))

    plot_accuracy(
        scale_factors,
        valid_acc,
        "Accuracy of User-based Filtering With Gender Column For Various Scale Factors",
        "knn_modified_gender_filtering",
    )

def plot_pred_premium(sparse_matrix, student_meta_data, val_data):
    normalized_premium_array = normalize_array(get_premium_array(student_meta_data))
    scale_factors = [0.0, 0.1, 0.5, 1.0, 2.0, 10.0]
    valid_acc = []
    for scale_factor in scale_factors:
        new_sparse_matrix = append_data_as_new_question(
            sparse_matrix, scale_array(normalized_premium_array, scale_factor)
        )
        valid_acc.append(knn_impute_by_user(new_sparse_matrix, val_data, 11))

    plot_accuracy(
        scale_factors,
        valid_acc,
        "Accuracy of User-based Filtering With Premium Column For Various Scale Factors",
        "knn_modified_premium_filtering",
    )   

# Order the students by the number of questions they have answered
# Return the user_id of these students and the number of questions they have answered
def order_students_by_num_questions(sparse_matrix):
    num_questions_answered = np.sum(~np.isnan(sparse_matrix), axis=1)
    sorted_indices = np.argsort(num_questions_answered)
    return sorted_indices, num_questions_answered[sorted_indices]

# Take the bottom 20% of students who have answered the least number of questions
# and filter the test_data to only include these students
def filter_data_to_lowest(test_data, sorted_indices):
    num_students = len(sorted_indices)
    num_students_to_filter = int(num_students * 0.2)
    students_to_filter = sorted_indices[:num_students_to_filter]
    filtered_test_data = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }
    for i in range(len(test_data["user_id"])):
        if test_data["user_id"][i] in students_to_filter:
            filtered_test_data["user_id"].append(test_data["user_id"][i])
            filtered_test_data["question_id"].append(test_data["question_id"][i])
            filtered_test_data["is_correct"].append(test_data["is_correct"][i])
    return filtered_test_data

def filter_data_to_partitions(test_data, sorted_indices, num_partitions):
    num_students = len(sorted_indices)
    partitioned_test_data = []
    partition_size = int(num_students / num_partitions)
    for i in range(num_partitions):
        partitioned_test_data.append({
            "user_id": [],
            "question_id": [],
            "is_correct": []
        })
        partition_start = i * partition_size
        partition_end = (i + 1) * partition_size
        students_to_filter = sorted_indices[partition_start:partition_end]
        for j in range(len(test_data["user_id"])):
            if test_data["user_id"][j] in students_to_filter:
                partitioned_test_data[i]["user_id"].append(test_data["user_id"][j])
                partitioned_test_data[i]["question_id"].append(test_data["question_id"][j])
                partitioned_test_data[i]["is_correct"].append(test_data["is_correct"][j])
    return partitioned_test_data

# Select only the students in the test data who have premium_pupil values that are not nan
def filter_data_to_premium(test_data, student_meta_data):
    filtered_test_data = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }
    for i in range(len(test_data["user_id"])):
        if student_meta_data["premium_pupil"][test_data["user_id"][i]] != None:
            filtered_test_data["user_id"].append(test_data["user_id"][i])
            filtered_test_data["question_id"].append(test_data["question_id"][i])
            filtered_test_data["is_correct"].append(test_data["is_correct"][i])
    return filtered_test_data

def full_modified_analysis():
    # Get the root of the project
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Update the paths to the data files
    sparse_matrix = load_train_sparse(os.path.join(project_root, "data")).toarray()
    val_data = load_valid_csv(os.path.join(project_root, "data"))
    test_data = load_public_test_csv(os.path.join(project_root, "data"))
    student_meta_data = load_student_csv(os.path.join(project_root, "data"))

    year_array = get_year_array(student_meta_data)
    year_sparse_matrix = append_data_as_new_question(sparse_matrix, year_array)
    plot_pred_year(year_array, year_sparse_matrix, val_data)

    month_array = get_month_array(student_meta_data)
    month_sparse_matrix = append_data_as_new_question(sparse_matrix, month_array)
    plot_pred_month(month_array, month_sparse_matrix, val_data)

    plot_pred_gender(sparse_matrix, student_meta_data, val_data)
    plot_pred_premium(sparse_matrix, student_meta_data, val_data)

    best_year_sparse_matrix = append_data_as_new_question(sparse_matrix, get_year_array(student_meta_data))
    best_year = knn_impute_by_user(best_year_sparse_matrix, test_data, 11)
    print("Best Year Accuracy: ", best_year)
    
    best_month_sparse_matrix = append_data_as_new_question(sparse_matrix, get_month_array(student_meta_data))
    best_month = knn_impute_by_user(best_month_sparse_matrix, test_data, 11)
    print("Best Month Accuracy: ", best_month)

    best_premium_sparse_matrix = append_data_as_new_question(sparse_matrix, get_premium_array(student_meta_data) * 1.0)
    best_prem = knn_impute_by_user(best_premium_sparse_matrix, test_data, 11)
    print("Best Premium Accuracy: ", best_prem)

    best_gender_sparse_matrix = append_data_as_new_question(sparse_matrix, get_gender_array(student_meta_data) * 0.5)
    best_gender = knn_impute_by_user(best_gender_sparse_matrix, test_data, 11)
    print("Best Gender Accuracy: ", best_gender)
    
    sparse_matrix = append_data_as_new_question(sparse_matrix, get_premium_array(student_meta_data) * 1.0)
    sparse_matrix = append_data_as_new_question(sparse_matrix, get_gender_array(student_meta_data) * 0.5)
    sparse_matrix = append_data_as_new_question(sparse_matrix, get_year_array(student_meta_data) * 0.1)
    combined_acc = knn_impute_by_user(sparse_matrix, test_data, 11)
    print("Combined Accuracy: ", combined_acc)

def main():
    # Get the root of the project
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Update the paths to the data files
    sparse_matrix = load_train_sparse(os.path.join(project_root, "data")).toarray()
    val_data = load_valid_csv(os.path.join(project_root, "data"))
    test_data = load_public_test_csv(os.path.join(project_root, "data"))
    student_meta_data = load_student_csv(os.path.join(project_root, "data"))

    sorted_indices, num_questions_answered = order_students_by_num_questions(sparse_matrix)
    premium_test_data = filter_data_to_premium(test_data, student_meta_data)
    premium_sparse = append_data_as_new_question(sparse_matrix, get_premium_array(student_meta_data) * 1)

    num_partitions = 5
    partitioned_test_data = filter_data_to_partitions(premium_test_data, sorted_indices, num_partitions)
    table_result_values = []
    for i in range(num_partitions):
        print('\n\n')
        print("Original on Partition ", i)
        og_acc = knn_impute_by_user(sparse_matrix, partitioned_test_data[i], 11)
        print("Premium on Partition ", i)
        new_acc = knn_impute_by_user(premium_sparse, partitioned_test_data[i], 11)
        table_result_values.append((i, og_acc, new_acc))

    # Create plot using table_result_values
    plt.figure(figsize=(10, 6))
    plt.plot([str(i) for i in range(num_partitions)], [val[1] for val in table_result_values], label="Original")
    plt.plot([str(i) for i in range(num_partitions)], [val[2] for val in table_result_values], label="Premium")
    plt.xlabel("Partition")
    plt.ylabel("Accuracy")
    plt.title("Accuracy of Original and Premium on Test Data with premium_pupil status Partitioned by Number of Questions Answered")
    plt.legend()
    plt.savefig("../Static/knn_modified_comparison_accuracy_premium_only.png", dpi=400)
    plt.show()


    print('\n\n')
    print("Original on All")
    knn_impute_by_user(sparse_matrix, premium_test_data, 11)
    print("Premium on All")
    knn_impute_by_user(premium_sparse, premium_test_data, 11)
    
    # Repeat everything above but this time using test_data instead of premium_test_data
    partitioned_test_data = filter_data_to_partitions(test_data, sorted_indices, num_partitions)
    table_result_values = []
    for i in range(num_partitions):
        print('\n\n')
        print("Original on Partition ", i)
        og_acc = knn_impute_by_user(sparse_matrix, partitioned_test_data[i], 11)
        print("Premium on Partition ", i)
        new_acc = knn_impute_by_user(premium_sparse, partitioned_test_data[i], 11)
        table_result_values.append((i, og_acc, new_acc))

    # Create plot using table_result_values

    plt.figure(figsize=(10, 6))
    plt.plot([str(i) for i in range(num_partitions)], [val[1] for val in table_result_values], label="Original")
    plt.plot([str(i) for i in range(num_partitions)], [val[2] for val in table_result_values], label="Premium")
    plt.xlabel("Partition")
    plt.ylabel("Accuracy")
    plt.title("Accuracy of Original and Premium on all Test Data Partitioned by Number of Questions Answered")
    plt.legend()
    plt.savefig("../Static/knn_modified_comparison_accuracy_all.png", dpi=400)
    plt.show()



if __name__ == "__main__":
    main()
