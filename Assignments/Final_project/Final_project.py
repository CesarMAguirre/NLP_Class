import pandas as pd

training_data_path = r"C:\BC\SP_25\NLP\NLP_Class\Assignments\Final_project\pii-detection-removal-from-educational-data\train.json"

training_data = pd.read_json(training_data_path, lines=True)
print(f'Shape of the training data: {training_data.shape}')
training_data.head(10)

