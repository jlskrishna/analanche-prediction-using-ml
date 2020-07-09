import numpy as np
from flask import Flask, request, jsonify, render_template
from joblib import load
app = Flask(__name__)
model = load('ava.save')

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/y_predict',methods=['POST'])
def y_predict():
    '''
    For rendering results on HTML GUI
    '''
    x_test = [[int(x) for x in request.form.values()]]
    if(x_test[0][0] == 0):
        x_test[0][0]=0
        x_test[0].insert(1,0)
    elif(x_test[0][0] == 1):
        x_test[0][0]=1
        x_test[0].insert(1,0)
    else:
        x_test[0][0]=0
        x_test[0].insert(1,1)
    print(x_test)
    prediction = model.predict(x_test)
    print(prediction)
    if(prediction[0]==0):
        output="high"
    elif(prediction[0]==1):
        output="low"
    elif(prediction[0]==2):
        output="moderate"
    
    return render_template('index1.html', prediction_text='Avalanche chances are {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.y_predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
