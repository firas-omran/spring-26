from sklearn.model_selection import train_test_split


SEX_MAP = {
    "male": 0.0,
    "female": 1.0,
}

EMBARKED_MAP = {
    "S": 0.0,
    "C": 1.0,
    "Q": 2.0,
}


def prepare_data(df):

    df = df.copy()

    df = df.drop(columns=[
        "PassengerId",
        "Name",
        "Ticket",
        "Cabin",
    ])

    y = df["Survived"].astype(int)
    X = df.drop(columns=["Survived"])

    # Keep missing values as NaN so the custom tree can route them probabilistically.
    X["Sex"] = X["Sex"].map(SEX_MAP)
    X["Embarked"] = X["Embarked"].map(EMBARKED_MAP)

    X = X.astype(float)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
