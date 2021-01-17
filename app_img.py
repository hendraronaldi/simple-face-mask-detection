import cv2
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.applications.vgg16 import VGG16 as PretrainedModel, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import joblib

@st.cache
def load_model():
	return tf.keras.models.load_model('model.h5')

@st.cache
def load_features():
	ptm = PretrainedModel(
	    include_top=False,
	    weights='imagenet',
	    input_shape=[200, 200] + [3]
	)
	dx = Flatten()(ptm.output)
	dm = Model(inputs=ptm.input, outputs=dx)
	return dm

@st.cache
def load_scaler():
	return joblib.load('scaler.pkl')

# sidebar
st.sidebar.header("Parameters")
threshold = st.sidebar.slider('Threshold', 0.0, 1.0, 0.5)

# main page
st.title("Face Mask Detection")
st.write("""
Source Notebook: [link](https://github.com/hendraronaldi/DPHI_data_science_challenges/blob/main/DPHI%20Face%20Mask%20Detection.ipynb)
""")

model = load_model()
dm = load_features()
scaler = load_scaler()

uploaded_img = st.file_uploader("Upload Image")
if uploaded_img is not None:
	try:
		file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
		frame = cv2.imdecode(file_bytes, 1)

		if frame.shape[0] > 500:
			st.image(frame, channels="BGR", width=500)
		else:
			st.image(frame, channels="BGR")

		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame = cv2.resize(frame, (200, 200))
		img = image.img_to_array(frame)
		img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
		img = preprocess_input(img)

		feat_test = dm.predict(img)
		feat_test = scaler.transform(feat_test)
		y_pred = 1-model.predict(feat_test)[0][0]
		if y_pred < threshold:
			st.subheader('Not using face mask!!!')
		else:
			st.subheader('Using face mask, good')
		st.text(f'Prediction Score: {y_pred}')
	except:
		st.subheader("Please upload image")
