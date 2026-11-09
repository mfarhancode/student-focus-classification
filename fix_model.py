
import h5py
import json
import shutil

# Copy original model
shutil.copy("keras_Model.h5", "keras_Model_fixed.h5")

# Fix old Keras configuration
with h5py.File("keras_Model_fixed.h5", "r+") as f:
  config = json.loads(f.attrs["model_config"])

  def fix_layers(obj):
    if isinstance(obj, dict):
      if obj.get("class_name") == "DepthwiseConv2D":
        obj.get("config", {}).pop("groups", None)

      for value in obj.values():
        fix_layers(value)

    elif isinstance(obj, list):
      for value in obj:
        fix_layers(value)

  fix_layers(config)

  f.attrs["model_config"] = json.dumps(config)

print("Model fixed successfully!")
