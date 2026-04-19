import argparse
import mlflow
import mlflow.sklearn
import json

import split_dataset
import featurization
import train
import evaluation
import dagshub


# =========================
# PIPELINE STEPS
# =========================

def split(input_path, output_train, output_test, test_size: float, seed: int):
    print(f"\n>>> Running split_data(test_size={test_size}, seed={seed})")

    train_df, test_df = split_dataset.split_data(
        input_path,
        output_train,
        output_test,
        test_size,
        seed
    )

    return train_df, test_df


def run_featurize(train_path, test_path, output_dir):
    print("\n>>> Running featurize()")
    return featurization.featurize(
        train_path,
        test_path,
        output_dir,
        "models/vectorizer.pkl"
    )


def run_train(train_data, model_out, seed: int):
    print(f"\n>>> Running train (LogReg only, seed={seed})")

    return train.train_model(
        train_data,
        model_out,
        seed
    )


def run_evaluate(model_path, eval_data, metrics_out):
    print("\n>>> Running evaluate()")

    return evaluation.evaluate(
        model_path,
        eval_data,
        metrics_out
    )


# =========================
# PIPELINE ORCHESTRATION
# =========================

def run_all(args):

    # =========================================================
    # MLflow + DAGSHUB TRACKING
    # =========================================================
    dagshub.init(repo_owner='NapoliFabian', repo_name='se4ai_mlops_project', mlflow=True)
    mlflow.set_experiment("BASELINE")

    with mlflow.start_run(run_name="logreg_pipeline"):

        # =========================
        # PATHS
        # =========================
        input_path = "data/raw/dataset.csv"
        train_csv = "data/interim/train.csv"
        test_csv = "data/interim/test.csv"

        features_dir = "data/processed"
        train_pkl = "data/processed/train.pkl"
        test_pkl = "data/processed/test.pkl"

        model_out = "models/model.pkl"
        metrics_out = "reports/metrics.json"

        # =========================
        # LOG PARAMS
        # =========================
        mlflow.log_params({
            "test_size": args.test_size,
            "seed": args.seed,
            "model_type": "logistic_regression"
        })

        # =========================
        # 1. SPLIT
        # =========================
        train_df, test_df = split(
            input_path,
            train_csv,
            test_csv,
            args.test_size,
            args.seed
        )

        mlflow.log_params({
            "n_train": len(train_df),
            "n_test": len(test_df)
        })

        # =========================
        # SAVE CLASS DISTRIBUTION
        # =========================
        class_dist = train_df["label"].value_counts(normalize=True).to_dict()

        with open("class_distribution.json", "w") as f:
            json.dump(class_dist, f, indent=2)

        mlflow.log_artifact("class_distribution.json")

        # =========================
        # 2. FEATURIZATION
        # =========================
        X_train, X_test, y_train, y_test = run_featurize(
            train_csv,
            test_csv,
            features_dir,
        )

        mlflow.log_param("n_features", X_train.shape[1])

        # =========================
        # 3. TRAIN (ONLY LOGISTIC REGRESSION)
        # =========================
        model = run_train(
            train_data=train_pkl,
            model_out=model_out,
            seed=args.seed
        )

        # =========================
        # 4. EVALUATE
        # =========================
        metrics = run_evaluate(
            model_path=model_out,
            eval_data=test_pkl,
            metrics_out=metrics_out
        )

        # =========================
        # MLflow logging
        # =========================
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(metrics_out)

        # =========================
        # SUMMARY
        # =========================
        print("\n" + "=" * 50)
        print("PIPELINE SUMMARY")
        print("=" * 50)

        print(f"Train size: {len(train_df)}")
        print(f"Test size : {len(test_df)}")

        print("\nMETRICS:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        print("=" * 50)

        return {
            "metrics": metrics
        }


# =========================
# CLI
# =========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=1234)

    args = parser.parse_args()

    run_all(args)


if __name__ == "__main__":
    main()