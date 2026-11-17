import pandas as pd
from scipy.stats import ks_2samp
from sklearn.datasets import load_iris


def check_data_drift():

    # 1. Load reference data
    iris = load_iris(as_frame=True)

    reference = iris.frame.sample(
        n=100, random_state=42
    )

    # 2. Simulate current production data
    current = iris.frame.sample(
        n=100, random_state=99
    ).copy()

    # Artificially introduce drift
    current['sepal length (cm)'] = (
        current['sepal length (cm)'] * 1.5
    )

    # 3. Detect drift using KS test
    drifted_columns = 0
    feature_cols = iris.feature_names

    print("\nData Drift Analysis")
    print("-------------------")

    for col in feature_cols:

        p_val = ks_2samp(
            reference[col],
            current[col]
        ).pvalue

        print(f"{col}: p-value = {p_val:.4f}")

        if p_val < 0.05:
            drifted_columns += 1

    # 4. Calculate dataset drift
    import pandas as pd
from scipy.stats import ks_2samp
from sklearn.datasets import load_iris


def check_data_drift():

    # 1. Load reference data
    iris = load_iris(as_frame=True)

    reference = iris.frame.sample(
        n=100, random_state=42
    )

    # 2. Simulate current production data
    current = iris.frame.sample(
        n=100, random_state=99
    ).copy()

    # 3. Introduce artificial data drift
    current['sepal length (cm)'] = (
        current['sepal length (cm)'] * 1.5
    )

    current['petal length (cm)'] = (
        current['petal length (cm)'] * 1.5
    )

    # 4. Detect drift using KS test
    drifted_columns = 0
    feature_cols = iris.feature_names

    print("\nData Drift Analysis")
    print("-------------------")

    results = []

    for col in feature_cols:

        p_val = ks_2samp(
            reference[col],
            current[col]
        ).pvalue

        drift = p_val < 0.05

        if drift:
            drifted_columns += 1

        print(
            f"{col}: p-value = {p_val:.4f}, "
            f"Drift = {drift}"
        )

        results.append({
            "Feature": col,
            "P_Value": p_val,
            "Drifted": drift
        })

    # 5. Calculate dataset drift
    dataset_drift = (
        drifted_columns >= len(feature_cols) / 2
    )

    print("\nDrifted Columns:", drifted_columns)

    print(
        "Dataset Drift Detected:",
        dataset_drift
    )

    # 6. Save results as CSV
    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "drift_results.csv",
        index=False
    )

    print("\nDrift results saved successfully!")

    return dataset_drift


if __name__ == "__main__":
    check_data_drift()