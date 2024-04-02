from utils import *
import numpy as np
import itertools
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import random

def sigmoid(x):
    """ Apply sigmoid function.
    """
    return 1 / (1 + np.exp(-x))

def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    
    log_lklihood = 0

    for user, question, c in zip(data["user_id"], data["question_id"], data["is_correct"]):
            mu = sigmoid((theta[user] - beta[question]))
            log_lklihood += c * np.log(mu) + (1-c)*np.log(1-mu)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################

    for i in range(len(data["user_id"])):

        user_id = data["user_id"][i]
        question_id = data["question_id"][i]
        is_correct = data["is_correct"][i]

        # Compute Sigmoid
        mu = sigmoid(theta[user_id] - beta[question_id])

        # Updates
        theta[user_id] += lr * (is_correct - mu)
        beta[question_id] -= lr * (is_correct - mu)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.random.randn(542)  # want this to have len number of students
    beta =  np.random.randn(1774) # want this to have len number of questions

    train_lld_lst = []
    val_lld_lst = []

    for i in range(iterations):
        train_neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        val_neg_lld = neg_log_likelihood(val_data, theta, beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        train_lld_lst.append(train_neg_lld)
        val_lld_lst.append(val_neg_lld)
        # print("NLLK: {} \t Score: {}".format(train_neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, score, train_lld_lst, val_lld_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])

def plot_log_likelihood(train_data, val_data):

    _, _, _, train_lst, val_lst = irt(train_data, val_data, 0.03, 20)
    
    plt.figure(figsize=(10,8))
    plt.subplots_adjust(hspace=0.5)  # Adjust vertical spacing between subplots
    plt.subplot(2,1,1)
    plt.plot(train_lst)
    plt.xlabel("Iteration")
    plt.ylabel("Negative Log Likelihood")
    plt.title("Train NLL over Iterations")

    plt.subplot(2,1,2)
    plt.plot(val_lst)
    plt.xlabel("Iteration")
    plt.ylabel("Negative Log Likelihood")
    plt.title("Val NLL over Iterations")

    plt.savefig("../Static/log_likelihood.png", dpi=400)
    plt.show()

def plot_probability_curves(train_data, val_data):

    q1, q2, q3 = 1, 2, 3
    theta, beta, _, _, _ = irt(train_data, val_data, 0.03, 20)

    q1_y_values = [sigmoid(theta[i] - beta[q1]) for i in range(len(theta))]
    q2_y_values = [sigmoid(theta[i] - beta[q2]) for i in range(len(theta))]
    q3_y_values = [sigmoid(theta[i] - beta[q3]) for i in range(len(theta))]

    plt.scatter(theta, q1_y_values, label=f"Question {q1} = {beta[q1]:.2f}")
    plt.scatter(theta, q2_y_values, label=f"Question {q2}= {beta[q2]:.2f}")
    plt.scatter(theta, q3_y_values, label=f"Question {q3}= {beta[q3]:.2f}")
    plt.title(f"Probability Distuestion for Question {1}")
    plt.xlabel("Theta")
    plt.ylabel("Probability of Correct Response")
    plt.legend()
    plt.savefig("../Static/probability_curve.png", dpi=400)

    plt.show()


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    
    lrs = [0.0001, 0.001, 0.01, 0.03, 0.05, 0.1, 0.5]
    num_iterations = [5, 10, 20, 50]

    results = {
        "lr": [],
        "iter": [],
        "val_acc": [],
        "theta": [],
        "beta": []
    }

    for item in tqdm(list(itertools.product(lrs, num_iterations)), desc="Hyperparameter Tuning"):

        lr, iter = item
        tqdm.write(f"Testing lr = {lr} and iteration = {iter}")
        results["lr"].append(lr)
        results["iter"].append(iter)

        theta, beta, val_acc, _, _ = irt(train_data, val_data, lr, iter)
        results["val_acc"].append(val_acc)
        results["theta"].append(theta)
        results["beta"].append(beta)
    
    idx = results["val_acc"].index(max(results['val_acc']))

    print(results["lr"][idx], results["iter"][idx], results["val_acc"][idx])

    ## Get the test accuracy
    theta_best, beta_best = results["theta"][idx], results["beta"][idx]
    print(f"The test accuracy is: {evaluate(test_data, theta_best, beta_best)}")

    plot_log_likelihood(train_data, val_data)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    plot_probability_curves(train_data, val_data)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    main()
