import numpy as np
from flask import Flask, request,render_template
import pickle
import pandas as pd

app=Flask(__name__)
model=pickle.load(open('grade_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    '''

    For rendering results on HTML GUI

    '''
    int_features = [int(x)for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    features_name=['school','sex','age','address','Pstatus','Medu','Fedu','Mjob',
                   'Fjob','reason','guardian','traveltime','studytime','failures',
                   'schoolsup','famsup','paid','activities','nursery','higher',
                   'internet','romantic','famrel','freetime','goout','Dalc',
                   'Walc','health','absences','G1','G2']
    
    df=pd.DataFrame(final_features,columns=features_name)
    prediction=model.predict(df)

    output=ceil(prediction[0])

    return render_template('index.html',prediction_text='The grade of the student is {}'.format(output))


if __name__=="_main__":
    app.run(debug=True)
    
