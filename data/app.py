from flask import Flask,render_template,request
from sklearn.linear_model import LinearRegression
import pandas as pd 
import  numpy as np
import pickle
from sklearn.preprocessing import PolynomialFeatures
app=Flask(__name__)
@app.route('/')
def home():
 return render_template('index.html')
@app.route('/predict',methods=['POST'])
def get_result():
 poly=pickle.load(open('poly1.pkl','rb'))
 model=pickle.load(open('model1.pkl','rb'))
 query=[[float(request.form['Experience'])]]
 X=poly.transform(query)
 sal=model.predict(X)
 return 'Dear'+request.form["name"]+'YourPredicted salary after'+request.form["Experience"]+'Experience is:'+str(sal);
if __name__=='__main__': 
 app.run(debug=True)
 