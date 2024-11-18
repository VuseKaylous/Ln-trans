from comet import download_model, load_from_checkpoint

# Choose your model from Hugging Face Hub
model_path = download_model("Unbabel/XCOMET-XL")
# or for example:
# model_path = download_model("Unbabel/wmt22-comet-da")

# Load the model checkpoint:
model = load_from_checkpoint(model_path)
