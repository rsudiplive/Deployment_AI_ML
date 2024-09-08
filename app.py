import pickle
from flask import Flask,app,jsonify,request,url_for,render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
#load the model
regmodel=pickle.load(open('regmodel.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_trans_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_trans_data)
    print(output[0])
    return jsonify([output[0]])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The House price prediction is {}".format(output))
    #It shows the home.html page and includes a message with the prediction result in it.


if __name__=="__main__":
    app.run(debug=True)


