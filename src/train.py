from __future__ import annotations

from pathlib import Path

from src.data import (
    load_december_data,
    load_prediction_template,
    load_train_data,
    load_validation_data,
)
from src.features import (
    DECEMBER_FEATURES,
    MAIN_FEATURES,
    add_lane_feature,
)
from src.models import (
    build_december_model,
    build_main_model,
)
from src.predict import (
    build_december_submission,
    build_validation_submission,
    save_submission,
)
from src.preprocessing import (
    apply_imputation,
    basic_cleaning,
    get_imputation_values,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TRAIN_PATH = RAW_DIR / "train-test.csv"
VALIDATION_PATH = RAW_DIR / "validation.csv"
TEMPLATE_PATH = RAW_DIR / "validation-predictions-template.csv"
DECEMBER_PATH = RAW_DIR / "december-chart-inputs.csv"


def main() -> None:
    print("Loading datasets...")

    train_raw = load_train_data(
        TRAIN_PATH
    )

    validation_raw = load_validation_data(
        VALIDATION_PATH
    )

    template = load_prediction_template(
        TEMPLATE_PATH
    )

    december_raw = load_december_data(
        DECEMBER_PATH
    )

    print(
        "Train:",
        train_raw.shape,
    )
    print(
        "Validation:",
        validation_raw.shape,
    )
    print(
        "December:",
        december_raw.shape,
    )

    # -------------------------------------------------
    # Common preprocessing
    # -------------------------------------------------

    train = basic_cleaning(
        train_raw
    )

    validation = basic_cleaning(
        validation_raw
    )

    december = basic_cleaning(
        december_raw
    )

    # Learn imputation values from labeled training data only
    imputation_values = get_imputation_values(
        train
    )

    train = apply_imputation(
        train,
        imputation_values,
    )

    validation = apply_imputation(
        validation,
        imputation_values,
    )

    december = apply_imputation(
        december,
        imputation_values,
    )

    # -------------------------------------------------
    # Main 12,000-row model
    # -------------------------------------------------

    print("\nTraining main Ridge model...")

    main_model = build_main_model(
        alpha=100.0
    )

    X_train = train[
        MAIN_FEATURES
    ].copy()

    y_train = train[
        "posted_rate"
    ].copy()

    X_validation = validation[
        MAIN_FEATURES
    ].copy()

    main_model.fit(
        X_train,
        y_train,
    )

    validation_predictions = (
        main_model.predict(
            X_validation
        )
    )

    validation_submission = (
        build_validation_submission(
            validation_df=validation,
            template_df=template,
            predictions=validation_predictions,
        )
    )

    validation_output_path = (
        OUTPUT_DIR
        / "validation_predictions_reproduced.csv"
    )

    save_submission(
        validation_submission,
        validation_output_path,
    )

    print(
        "Saved reproduced validation predictions:",
        validation_output_path,
    )

    # -------------------------------------------------
    # Reduced December model
    # -------------------------------------------------

    print("\nTraining December Ridge model...")

    december_train = add_lane_feature(
        train
    )

    december_input = add_lane_feature(
        december
    )

    december_model = build_december_model(
        alpha=100.0
    )

    X_december_train = december_train[
        DECEMBER_FEATURES
    ].copy()

    y_december_train = december_train[
        "posted_rate"
    ].copy()

    X_december = december_input[
        DECEMBER_FEATURES
    ].copy()

    december_model.fit(
        X_december_train,
        y_december_train,
    )

    december_predictions = (
        december_model.predict(
            X_december
        )
    )

    december_submission = (
        build_december_submission(
            december_df=december_input,
            predictions=december_predictions,
        )
    )

    december_output_path = (
        OUTPUT_DIR
        / "december_chart_inputs_reproduced.csv"
    )

    save_submission(
        december_submission,
        december_output_path,
    )

    print(
        "Saved reproduced December predictions:",
        december_output_path,
    )

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()