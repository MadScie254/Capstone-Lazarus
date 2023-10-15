import streamlit as st
import pandas as pd
import tensorflow 

data = { 0: 'Corn (maize) Status:Cercospora leaf spot',
                    1: 'Corn (maize) Status:Common Rust',
                    2: 'Corn (maize) Status:Northern Leaf Blight',
                    3: 'Corn (maize) Status:Northern Leaf Blight',
                    4: 'Corn (maize) Status:Northern Leaf Blight',
                    5: 'Corn (maize) Status: Healthy',
                    6: 'Potato Status:Early Blight',
                    7: 'Potato Status:Late Blight',
                    8: 'Potato Status:Healthy',
                    9: 'Tomato Status:Bacterial Spot',
                    10: 'Tomato Status:Early Blight',
                    11: 'Tomato Status:Late Blight',
                    12: 'Tomato Status:Leaf Mold',
                    13: 'Tomato Status:Septoria leaf spot',
                    14: 'Tomato Status:Spider mites Two spotted spider mite',
                    15: 'Tomato Status:Target Spot',
                    16: 'Tomato Status:Tomato Yellow Leaf Curl Virus',
                    17: 'Tomato Status:Tomato mosaic virus',
                    18: 'Tomato Status:Healthy'}

df = pd.DataFrame(list(data.values()), columns=['Crop_Status'])
df['Status'] = df['Crop_Status'].apply(lambda x: x.split(':')[-1].strip())
df['Name'] = df['Crop_Status'].apply(lambda x: x.split(':')[0].replace('Status', '').strip())

# Extract 'Status' by splitting on ':' and taking the second part,
# then remove any leading or trailing spaces
df['Status'] = df['Crop_Status'].apply(lambda x: x.split(':')[-1].strip())

df = df[['Name', 'Status']]

#title
st.title('Leaf Classification')

#header
st.header('Please Upload an image of a Leaf')

#upload file
file = st.file_uploader('', type= ['jpeg', 'jpg', 'png'])

#load classifier
from tensorflow.keras.models import load_model
model = load_model('./inception_lazarus')

from PIL import Image, ImageOps
import numpy as np

def import_and_predict(image_data, model):
    size = (256,256)
    image = ImageOps.fit(image_data,size,Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)

    return prediction

if file is None:
    st.text('Please upload an image file')

else:
    image = Image.open(file)
    st.image(image,use_column_width=True)
    predictions = import_and_predict(image,model)
    class_names = df
    outs = 'This is a '+class_names.iloc[np.argmax(predictions)]['Name'] +',' + ' Status is '+class_names.iloc[np.argmax(predictions)]['Status']
    st.success(outs)

