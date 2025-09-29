from src.train import read_data_config
import torch
from src.model import FeatureTokenizerTransformer

data = "ltt"
best_path = "artifacts/ltt-normal-rex-finetuning/ltt-normal-rex-finetuning-best.ckpt"
scaler_save_dir = "artifacts/ltt-normal-rex-finetuning"

cont_features, cat_features, labels = read_data_config(f"configs/{data}.json")

# Create minimal sample inputs for ONNX export
n_cont_features = len(cont_features)
n_cat_features = len(cat_features)

# Load best model for export
best_model = FeatureTokenizerTransformer.load_from_checkpoint(best_path, map_location='cpu')
best_model.eval()
best_model = best_model.cpu()  # Ensure model is on CPU

# Sample inputs (batch_size=1) - ensure they are on CPU
sample_X_cont = torch.randn(1, n_cont_features, dtype=torch.float32).cpu()
sample_X_cat = torch.zeros(1, n_cat_features, dtype=torch.long).cpu()

# Export to ONNX
torch.onnx.export(
    best_model,
    (sample_X_cont, sample_X_cat),
    f"{scaler_save_dir}/final_model.onnx",
    input_names=['continuous_features', 'categorical_features'],
    output_names=['predictions']
)
print(f"ONNX model saved to {scaler_save_dir}/final_model.onnx")