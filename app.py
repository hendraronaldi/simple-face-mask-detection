import cv2
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.applications.vgg16 import VGG16 as PretrainedModel, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import joblib

st.title("Face Mask Detection")
st.write("""
Source Notebook: [link](https://github.com/hendraronaldi/DPHI_data_science_challenges/blob/main/DPHI%20Face%20Mask%20Detection.ipynb)
""")
st.write("""
Dataset: [link](https://drive.google.com/file/d/1_W2gFFZmy6ZyC8TPlxB49eDFswdBsQqo/view?usp=sharing)
""")
run = st.checkbox('Run Webcam')
FRAME_WINDOW = st.image([])
# camera = cv2.VideoCapture(0)
isCameraOn = False
camera = None
# st.subheader('Face Mask Prediction')
subH = st.empty()
result = st.empty()
status = st.empty()

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

model = load_model()
dm = load_features()
scaler = load_scaler()

while run:
	status.write('')
	camera = cv2.VideoCapture(0)
	if not camera.isOpened():
		camera.open()

	subH.subheader('Face Mask Prediction')
	ret, frame = camera.read()
	if ret:
		isCameraOn = True
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame = cv2.flip(frame, 1)
		FRAME_WINDOW.image(frame)
		frame = cv2.resize(frame, (200, 200))
		img = image.img_to_array(frame)
		img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
		img = preprocess_input(img)

		feat_test = dm.predict(img)
		feat_test = scaler.transform(feat_test)
		y_pred = np.round(model.predict(feat_test)[0][0])
		if y_pred == 1:
			result.text('Not using mask!!!')
		else:
			result.text('Using Mask, Good')


if isCameraOn:
	isCameraOn = False
	camera.release()
	cv2.destroyAllWindows()