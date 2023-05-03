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

        scaler_RMF = pickle.load(open('scaler_RMF.pkl','rb'))
        transformed = scaler_RMF.transform(np.asarray([k[1],k[2],k[3]]).reshape(1, -1))
        
        asdasd = pickle.load(open('kmeans_M.pkl','rb'))
        K0= asdasd.predict(transformed[0][0].reshape(-1, 1))

        asdasd = pickle.load(open('kmeans_F.pkl','rb'))
        K1 = asdasd.predict(transformed[0][1].reshape(-1, 1))

        asdasd = pickle.load(open('kmeans_R.pkl','rb'))
        K2 = asdasd.predict(transformed[0][2].reshape(-1, 1))

        
        total = K0 + K1 + K2 
        if(total > 4 ):         return render_template('index.html', prediction_text=('cluster is {}'.format(cluster) , 'with SEGMENTATION = HIGH' , 'with your input is {} '.format(k)))
        elif(total < 2 ):       return render_template('index.html', prediction_text=('cluster is {}'.format(cluster) , 'with SEGMENTATION = LOW' , 'with your input is {} '.format(k)))
        elif( 2 <= total <=4 ): return render_template('index.html', prediction_text=('cluster is {}'.format(cluster) , 'with SEGMENTATION = MID' , 'with your input is {} '.format(k)))
        return render_template('index.html', prediction_text=('cluster is {}'.format(cluster) ,'with your input is {} '.format(k)))
        
    # except:
        # return render_template('index.html', prediction_text='Please give correct input, all either 0 or 1 (except age b/w 5:100) $ {} '.format(codn))


if __name__ == "__main__":
   app.run(debug=True)

# app.run(host='localhost',port=80)
