from google.colab import drive
drive.mount("/content/gdrive")

from imageai.Prediction.Custom import CustomImagePrediction
import os

execution_path = os.getcwd()

prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath("gdrive/My Drive/autonomous rc car project/idenprof-jpg/idenprof/models/model_ex-200_acc-1.000000.h5")
prediction.setJsonPath("gdrive/My Drive/autonomous rc car project/idenprof-jpg/idenprof/json/model_class.json")
prediction.loadModel(num_objects=2)

predictions, probabilities = prediction.predictImage("gdrive/My Drive/autonomous rc car project/3.jpg", result_count=2)

print(predictions[0])
print(probabilities[0])

print(predictions[1])
print(probabilities[1])
