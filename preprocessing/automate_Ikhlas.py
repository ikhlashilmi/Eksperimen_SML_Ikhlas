import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)

    X = df.drop("price", axis=1)
    y = df["price"]

    cat_cols = X.select_dtypes(include="object").columns
    num_cols = X.select_dtypes(exclude="object").columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first"), cat_cols)
        ]
    )

    X_processed = preprocessor.fit_transform(X)

    feature_names = (
        num_cols.tolist() +
        preprocessor.named_transformers_["cat"]
        .get_feature_names_out(cat_cols).tolist()
    )

    df_processed = pd.DataFrame(X_processed, columns=feature_names)
    df_processed["price"] = y.values

    df_processed.to_csv(output_path, index=False)

if __name__ == "__main__":
    preprocess_data(
        "../housing_raw/housing.csv",
        "housing_preprocessed.csv"
    )
