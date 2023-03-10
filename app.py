

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

#import image
from tensorflow.keras.preprocessing import image
#import opencv
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://e0.pxfuel.com/wallpapers/155/795/desktop-wallpaper-cartoon-seamless-pattern-in-pastel-pink-summer-fruit-background-girly-vector-2088365-vector-art-at-vecteezy.jpg");
background-size: 180%;
background-position: top;
background-repeat: no-repeat;
background-attachment: local;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)


st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache(allow_output_mutation=True)
def load_model():
	model = tf.keras.models.load_model('./convolutional_2D_pomegranant.h5')
	return model


def predict_class(image, model):
    
    image = tf.cast(image, tf.float32)
    #image = image.img_to_array(image)
   # image=cv2.imread(image)
    image = np.expand_dims(image, axis = 0)
    prediction = model.predict(image)
    return prediction


model = load_model()
st.title('Pomegranate Disease Prediction')

file = st.file_uploader("Upload an image of a Pomegranate")

if file is None:
	st.text('Waiting for upload....')

else:
	slot = st.empty()
	slot.text('Running inference....')

	test_image = tf.keras.preprocessing.image.load_img(file,target_size=(128,128))
    
    
    

	st.image(test_image, caption="Input Image", width = 200)

	pred = predict_class(np.asarray(test_image), model)

	class_names =["Stage-1","Stage-2","Stage-3","Good"]

	result = class_names[np.argmax(pred)]
	if result=="Good":
		 output = 'The Furit is Good '

	else:	
	    output = 'The Disease is ' + result

	slot.text('Done')

	st.success(output)
