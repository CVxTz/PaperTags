from pathlib import Path

import fire
import joblib
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def split_data(data_path, target_path):
    data = pd.read_json(data_path)
    data["tasks"] = data.tasks.apply(lambda x: [task for task in x if task])
    data = data[(data.tasks.str.len() > 0)]

    mlb = MultiLabelBinarizer()
    mlb.fit(data["tasks"])

    train = data[data.date.dt.year < 2023]
    val = data[data.date.dt.year.isin((2023,))]

    joblib.dump(mlb, Path(target_path) / "mlb.joblib")

    train.to_json(
        Path(target_path) / "train.json", orient="records", indent=2, force_ascii=False
    )
    val.to_json(
        Path(target_path) / "val.json", orient="records", indent=2, force_ascii=False
    )


if __name__ == "__main__":
    fire.Fire(split_data)
