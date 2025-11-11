import torch
import numpy as np
from src.model import FeatureTokenizerTransformer
import pandas as pd
import joblib

experiment_name = "quad-normal-hyperparam-snake"
encoder = "quad"

ckpt_path = f"artifacts/{experiment_name}/{experiment_name}-best-v82.ckpt"
model = FeatureTokenizerTransformer.load_from_checkpoint(ckpt_path, map_location=torch.device("cpu"))

model.eval()
model.to("cpu")
num = 200

cont_base = np.linspace(-10, 10, num).reshape(-1, 1).astype(np.float32)

cont_features = cont_base

cont_raw = cont_features.copy()
cat_features = np.ones((num, 1), dtype=np.int64)

feature_scaler = joblib.load(f"artifacts/{encoder}_encoders/feature_scaler.pkl")
label_scaler = joblib.load(f"artifacts/{encoder}_encoders/label_scaler.pkl")
cat_encoders = joblib.load(f"artifacts/{encoder}_encoders/cat_encoders.pkl")

cont_features = feature_scaler.transform(cont_features)

for col, label_encoder in cat_encoders.items():
    cat_features[:, 0] = label_encoder.transform(cat_features[:, 0])

cont_t = torch.from_numpy(cont_features)
cat_t = torch.from_numpy(cat_features)

with torch.no_grad():
	result = model(cont_t, cat_t)
	
result = result.numpy()
result = label_scaler.inverse_transform(result)

cont_raw = np.asarray(cont_raw)
n_cont_dims = cont_raw.shape[1] if cont_raw.ndim > 1 else 1
cont_cols = [f"cont_{i}" for i in range(n_cont_dims)]
df_cont = pd.DataFrame(cont_raw.reshape(cont_raw.shape[0], -1), columns=cont_cols)
df_pred = pd.DataFrame(result.reshape(result.shape[0], -1), columns=["Prediction"])
df = pd.concat([df_cont, df_pred], axis=1)
df.to_csv(f"artifacts/{experiment_name}/predictions.csv", index=False)