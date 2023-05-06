import re
import numpy as np
import os
from flask import Flask, app,request,render_template, flash
from tensorflow.keras import models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.python.ops.gen_array_ops import concat
from tensorflow.keras.applications.inception_v3 import InceptionV3,preprocess_input
import requests
from flask import Flask, request, render_template, redirect, url_for , Markup

#Loading the model



model = load_model(r"InceptionV3-covid.h5")
app = Flask(__name__)

app = Flask(__name__)

# Set the secret key to some random bytes. Keep this really secret!
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


@app.route('/')
def index():
	return render_template('demo1.html')


@app.route('/home')
def home():
	return render_template("demo1.html")



@app.route('/result',methods=["GET","POST"])
def result():
	if request.method=="POST":
		f = request.files['i1']
		basepath = os.path.dirname(__file__)
		#print("current path",basepath)
		filepath = os.path.join(basepath,'uploads',f.filename)
		#print("upload folder is",filepath)
		f.save(filepath)

		img = image.load_img(filepath,target_size=(299,299))
		x = image.img_to_array(img)
		x = np.expand_dims(x,axis = 0)
		img_data = preprocess_input(x)
		prediction= np.argmax(model.predict(img_data),axis = 1)

		index = ['Normal','Covid19','Viral Pneumonia']
		#result = str(index[output[0]])
		result = str(index[prediction[0]])
		#flash(result)
		#print(result)
		message = Markup(result)
		flash(message)
		return render_template('output.html')



"""Running our application"""
if __name__=="__main__":
	app.run(debug=True)



