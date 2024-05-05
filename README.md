# Ex.No: 13  Machine Learning – Mini Project  
## DATE: 22/4/2024                                                                           
## REGISTER NUMBER : 212221040084
# AIM: 
To write a program to train the classifier for Diabetes.
#  Algorithm:
Step 1: Import packages.
Step 2: Get the data.
Step 3: Split the data.
Step 4: Scale the data.
Step 5: Instantiate model.
Step 6: Create Gradio Function.
Step 7: Print Result.
# Program:
```
import numpy as np
import pandas as pd
pip install gradio
pip install typing-extensions --upgrade
pip install --upgrade typing
pip install typing-extensions --upgrade
import gradio as gr
data = pd.read_csv('/content/diabetes.csv')
data.head()
print(data.columns)
x = data.drop(['Outcome'], axis=1)
y = data['Outcome']
print(x[:5])
#split data
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test= train_test_split(x,y)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

from sklearn.neural_network import MLPClassifier
model = MLPClassifier(max_iter=1000, alpha=1)
model.fit(x_train, y_train)
print("Model Accuracy on training set:", model.score(x_train, y_train))
print("Model Accuracy on Test Set:", model.score(x_test, y_test))

def diabetes(Pregnancies, Glucose, Blood_Pressure, SkinThickness, Insulin, BMI,Diabetes_Pedigree, Age):
    x = np.array([Pregnancies,Glucose,Blood_Pressure,SkinThickness,Insulin,BMI,Diabetes_Pedigree,Age])
    prediction = model.predict(x.reshape(1, -1))
    if(prediction==0):
      return "NO"
    else:
      return "YES"

outputs = gr.Textbox()
app = gr.Interface(fn=diabetes, inputs=['number','number','number','number','number','number','number','number'], outputs=outputs,description="Detection of Diabeties")
app.launch(share=True)
```

# Output:
![WhatsApp Image 2024-04-22 at 11 29 01_4f22b547](https://github.com/HibaRajarajeswari/DIABETES-CLASSIFIER/assets/129970809/f424c4d6-6c97-4e06-bd97-e2dc888cf7b1)
![WhatsApp Image 2024-04-22 at 11 29 59_18e014f5](https://github.com/HibaRajarajeswari/DIABETES-CLASSIFIER/assets/129970809/5c569a5a-27a0-4523-a5c8-396977e73ea5)
![WhatsApp Image 2024-04-22 at 11 30 25_0baa9fe3](https://github.com/HibaRajarajeswari/DIABETES-CLASSIFIER/assets/129970809/e84dd75e-408e-4fac-93ce-02d003ffeb91)



# Result:
Thus the system was trained successfully and the prediction was carried out.
