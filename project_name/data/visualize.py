import os
import pyminiply

models_dir = r"C:\Users\ozdep\Documents\aml\Applied-ML-Group8\project_name\models"
model_file = os.path.join(models_dir, "obj_01.ply")

mesh = pyminiply.read_as_mesh(model_file)
mesh.plot()
