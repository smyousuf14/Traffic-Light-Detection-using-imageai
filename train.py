from imageai.Prediction.Custom import ModelTraining
from google.colab import drive

drive.mount("/content/gdrive")

model_trainer = ModelTraining()
model_trainer.setModelTypeAsResNet()
model_trainer.setDataDirectory("gdrive/My Drive/autonomous rc car project/idenprof-jpg/idenprof")
model_trainer.trainModel(num_objects = 2, num_experiments = 200, enhance_data = True, batch_size=32, show_network_summary= True)
