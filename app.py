import numpy as np
from flask import Flask,request,render_template
import pickle

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home ():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict() :
    int_features = [int(x) for x in request.form.values()]
    features=[np.array(int_features)]
    prediction=model.predict(features)
    
    output=round(prediction[0],2)
    
    return render_template('index.html',prediction_text="Student Performance with student's overall final grade  is {}".format(output))

if __name__ == '__main__':
    app.run()
