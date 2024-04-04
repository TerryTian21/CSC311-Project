from sklearn.impute import KNNImputer
from sklearn.metrics.pairwise import cosine_similarity
from utils import *
import numpy as np
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


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


def main():
    # Get the root of the project
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Update the paths to the data files
    sparse_matrix = load_train_sparse(os.path.join(project_root, "data")).toarray()
    val_data = load_valid_csv(os.path.join(project_root, "data"))
    test_data = load_public_test_csv(os.path.join(project_root, "data"))
    student_meta_data = load_student_csv(os.path.join(project_root, "data"))
    get_year_array(student_meta_data)
    get_month_array(student_meta_data)

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
