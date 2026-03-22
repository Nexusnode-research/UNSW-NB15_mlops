import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

class CategoricalPreprocessor:
    """Wraps an OrdinalEncoder with identical settings used in the notebook."""
    def __init__(self, cat_cols):
        self.cat_cols = list(cat_cols)
        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    def fit(self, df: pd.DataFrame):
        self.encoder.fit(df[self.cat_cols].astype(str))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[self.cat_cols] = self.encoder.transform(out[self.cat_cols].astype(str))
        return out
