import argparse
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import json

import split_dataset
import featurization
import train
import evaluation
import dagshub


# PIPELINE STEPS

def split(input_path, output_train, output_test, test_size: float, seed: int):
    print("\n>>> Running split_data")

    return split_dataset.split_data(
        input_path,
        output_train,
        output_test,
        test_size,
        seed
    )


def run_featurize(train_path, test_path, output_dir, method):
    print("\n>>> Running featurize()")

    return featurization.featurize(
        train_path,
        test_path,
        output_dir,
        vectorizer_path="models/vectorizer.pkl",
        method=method
    )


def run_train(train_data, model_out, seed: int, model_type):
    print("\n>>> Running train")

    return train.train_model(
        train_data,
        model_out,
        seed,
        model_type=model_type
    )


def run_evaluate(model_path, eval_data, metrics_out):
    print("\n>>> Running evaluate()")

    return evaluation.evaluate(
        model_path,
        eval_data,
        metrics_out
    )


# ORCHESTRATION

def run_all(args):

    dagshub.init(
        repo_owner="NapoliFabian",
        repo_name="se4ai_mlops_project",
        mlflow=True
    )

    mlflow.set_experiment("FakeNews_Classification")

    with mlflow.start_run(run_name=f"{args.model_type}_pipeline"):

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
        # PARAMS
        # =========================
        mlflow.log_params({
            "test_size": args.test_size,
            "seed": args.seed,
            "model_type": args.model_type
        })

        # =========================
        # SPLIT
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

        mlflow.log_artifact("class_distribution.json")

        # =========================
        # FEATURIZATION (DEPEND ON MODEL)
        # =========================
        if args.model_type == "logreg":
            method = "tfidf"
        else:
            method = "sbert"

        X_train, X_test, y_train, y_test = run_featurize(
            train_csv,
            test_csv,
            features_dir,
            method=method
        )

        mlflow.log_param("embedding", method)

        # =========================
        # TRAIN
        # =========================
        model, loss_history = run_train(
            train_data=train_pkl,
            model_out=model_out,
            seed=args.seed,
            model_type=args.model_type
        )

        # =========================
        # EVALUATION
        # =========================
        metrics = run_evaluate(
            model_path=model_out,
            eval_data=test_pkl,
            metrics_out=metrics_out
        )

        # =========================
        # LOG MODEL
        # =========================
        if args.model_type == "logreg":
            mlflow.sklearn.log_model(model, "model")
        else:
            mlflow.pytorch.log_model(model, "model")

        mlflow.log_metrics(metrics)
        mlflow.log_artifact(metrics_out)

        # =========================
        # SUMMARY
        # =========================
        print("\n" + "=" * 50)
        print("PIPELINE SUMMARY")
        print("=" * 50)

        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        print("=" * 50)

        return {"metrics": metrics}


# =========================
# CLI
# =========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=1234)

    # 🔥 NEW PARAM
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["logreg", "nn"],
        default="nn",
        required=False
        
    )

    args = parser.parse_args()

    run_all(args)


if __name__ == "__main__":
    main()