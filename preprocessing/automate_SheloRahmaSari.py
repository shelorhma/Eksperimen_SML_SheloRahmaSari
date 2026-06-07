import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

RAW_DATA = "../namadataset_raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_DIR = "dataset_preprocessing"
OUTPUT_FILE = f"{OUTPUT_DIR}/telco_clean.csv"


def preprocess():
    df = pd.read_csv(RAW_DATA)

    # Hapus customerID
    df = df.drop("customerID", axis=1)

    # Konversi TotalCharges
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Hapus missing value
    df = df.dropna()

    # Encoding target
    df["Churn"] = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    # One Hot Encoding
    categorical_columns = df.select_dtypes(
        include="object"
    ).columns

    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        drop_first=True
    )

    # Scaling
    scaler = StandardScaler()

    numerical_columns = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    df[numerical_columns] = scaler.fit_transform(
        df[numerical_columns]
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("Preprocessing selesai")
    print("Shape:", df.shape)


if __name__ == "__main__":
    preprocess()