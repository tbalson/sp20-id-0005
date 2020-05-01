# *project-code* directory

### This directory holds the files python files necessary for training the model, receiving a file from flask upload, and returning a prediction for said file.

## gatherData.py
This file includes methods that receive a file, and make a prediction.
### Functions:
#### upload()
Receives the file uploaded from the flask request. Saves this file to the directory */project-code/user_input_files*

Calls the funciton **predict()** from the file trainer.py to acquire a prediction from the model. Finally, it opens a confusion matrix for the model that is saved in */project-code/savedFiles* and sends the prediction, confusion matrix, and other variables back to the web page via **render_template()**

#### findScores()
Opens the previously saved confusion matrix for the model and calculates accuracy, precision, recall, and f1 score from these values and returns these values.


