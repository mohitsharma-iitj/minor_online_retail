from ast import If
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# load the model from disk
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    # try:
        recieved = [x for x in request.form.values()]
        k = np.array(recieved)  #final_features
        
        vectorizer = pickle.load(open('transform_description_name_to_predict_cluster.pkl','rb'))
        asdfhklj = vectorizer.transform([k[0]])

        model_for_cluster = pickle.load(open('model_for_cluster.pkl','rb'))
        cluster = model_for_cluster.predict(asdfhklj)

        
        return render_template('index.html', prediction_text=('cluster is {}'.format(cluster) ,'with your input is {} '.format(k)))
        
    # except:
        # return render_template('index.html', prediction_text='Please give correct input, all either 0 or 1 (except age b/w 5:100) $ {} '.format(codn))


if __name__ == "__main__":
    app.run(debug=True)

#app.run(host='localhost',port=80)
