import os
from pathlib import Path
import sys

data_dir = os.path.join("e:\\Devs\\train-transformer-with-books-202605\\data\\")
print(os.listdir(data_dir))
save_dir = Path(__file__).resolve().parents[2] / "data"
print(os.listdir(save_dir))
