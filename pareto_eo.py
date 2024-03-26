from itertools import product

from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch

from fairlearn.datasets import fetch_adult
from fairlearn.postprocessing import ThresholdOptimizer


def get_pareto_front(costs):
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(
                costs[is_efficient] < c, axis=1
            )  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient


def main():
    # %%
    # Below we load the "Adult" census dataset and split its features, sensitive
    # features, and labels into train and test sets.

    data = fetch_adult()
    X_raw = data.data
    y = (data.target == ">50K") * 1

    # Need to cast categorical columns to integer for sklearn to digest
    categorical_columns = X_raw.columns[X_raw.dtypes == "category"]
    for column in categorical_columns:
        X_raw[column] = X_raw[column].cat.codes

    # A is the protected attribute
    A = X_raw["sex"]

    # 60-20-20 Train-Valid-Test split
    (X_train, X_not_train, y_train, y_not_train, A_train, A_not_train) = (
        train_test_split(X_raw, y, A, test_size=0.4, random_state=12345, stratify=y)
    )
    (X_valid, X_test, y_valid, y_test, A_valid, A_test) = train_test_split(
        X_not_train,
        y_not_train,
        A_not_train,
        test_size=0.5,
        random_state=12345,
        stratify=y_not_train,
    )

    X_train = X_train.reset_index(drop=True)
    X_valid = X_valid.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_valid = y_valid.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    A_train = A_train.reset_index(drop=True)
    A_valid = A_valid.reset_index(drop=True)
    A_test = A_test.reset_index(drop=True)

    pareto_preds = run_eo(
        X_train, y_train, X_valid, y_valid, X_test, y_test, A_train, A_valid, A_test
    )


def run_eo(
    X_train, y_train, X_valid, y_valid, X_test, y_test, A_train, A_valid, A_test
):
    # Train base CLF on X_train, y_train
    clf = LogisticRegression(solver="liblinear", fit_intercept=True)
    clf.fit(X_train, y_train)
    threshold_optimizer = ThresholdOptimizer(
        estimator=clf,
        objective="accuracy_score",
        constraints="demographic_parity",
        predict_method="predict_proba",
        grid_size=1000,  # Lower this for much faster computation of PF
        prefit=True,
    )

    # Fit EqualizedOdds on X_valid, y_valid
    threshold_optimizer.fit(X_valid, y_valid, sensitive_features=A_valid)

    # Compute Pareto front
    groups = A_valid.unique()
    # Scaled accuracies are for efficient computing of overall accuracy (simply sum over scaled accs)
    scaled_accuracies = {
        group: threshold_optimizer._tradeoff_curve[group]["y"]
        * A_valid.value_counts()[group]
        / len(A_valid)
        for group in groups
    }
    # Per-group metrics, e.g. tpr, ppr
    # Note that we directly take "x" here because the target value in the algorithm is
    # reached with only small numerical instabilities possible due to the convex hull
    # interpolation employed
    #
    # E.g. if the algorithm has "x" set to 0.6, then it will find good combinations of
    # (p0, cond0, p1, cond1) that reach the "x" value for the target metric for this group
    # with very little numerical error (e.g. 0.5993)
    target_metrics = {
        group: threshold_optimizer._tradeoff_curve[group]["x"] for group in groups
    }
    # Look at all combinations of the threshold indexs considered per group (1001 ** N_groups by default)
    idx_combinations = list(
        product(range(len(threshold_optimizer._tradeoff_curve[0])), repeat=len(groups))
    )

    # Add all submodels to all_submodels dict
    all_submodels = {}
    for idx_combination in tqdm(idx_combinations):
        idx_per_group = {group: idx for group, idx in zip(groups, idx_combination)}
        target_metric_per_group = [
            target_metrics[group][idx] for group, idx in idx_per_group.items()
        ]
        disparity = max(target_metric_per_group) - min(target_metric_per_group)
        accuracy = sum(
            [scaled_accuracies[group][idx] for group, idx in idx_per_group.items()]
        )

        all_submodels[frozenset(idx_per_group.items())] = [disparity, accuracy]

    scores = np.array(list(all_submodels.values()))
    scores[:, 1] = -scores[
        :, 1
    ]  # Need to invert accuracy scores so both metrics are to be minimized

    # Compute validation set Pareto Front
    pareto_front = get_pareto_front(scores)
    pareto_front_idxs = np.array(list(all_submodels.keys()))[pareto_front]

    # Extract interpolation dicts that are part of the Pareto front for inferrence
    interpolation_dicts = []
    for idxs in pareto_front_idxs:
        interpolation_dict = {}
        for sensitive_feature_value, idx in idxs:
            best_interpolation = threshold_optimizer._tradeoff_curve[
                sensitive_feature_value
            ].transpose()[idx]
            interpolation_dict[sensitive_feature_value] = Bunch(
                p0=best_interpolation.p0,
                operation0=best_interpolation.operation0,
                p1=best_interpolation.p1,
                operation1=best_interpolation.operation1,
            )
        interpolation_dicts.append(interpolation_dict)

    ### INFERRENCE ###
    # Keep copy of original interpolation dict
    orig_interpolation_dict = (
        threshold_optimizer.interpolated_thresholder_.interpolation_dict
    )
    # Compute predictions for every interpolation dict on the validation set's Pareto front
    pareto_preds = []
    for interpolation_dict in interpolation_dicts:
        threshold_optimizer.interpolated_thresholder_.interpolation_dict = (
            interpolation_dict
        )
        pareto_preds.append(
            threshold_optimizer.predict(X_test, sensitive_features=A_test)
        )
    # Reput original interpolation dict
    threshold_optimizer.interpolated_thresholder_.interpolation_dict = (
        orig_interpolation_dict
    )

    # This is a list of size [n_items_on_pareto_front, n_test_items]
    # You can now compute any fairness, accuracy metric from these, y_test, and A_test
    return pareto_preds


if __name__ == "__main__":
    main()
