import numpy as np
import cv2
import seaborn as sns
import tensorflow as tf
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt


#On charge notre modèle CNN entrainer
@st.cache_resource
def load_model_CNN():
    model = tf.keras.models.load_model("cnn_model_cifar10.keras")
    return model

model = load_model_CNN()

#Définir les classes CIFAR10
cifar10_class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


st.markdown("<h1 style='text-align: center;'>CIFAR 10 Classifier</h1>", unsafe_allow_html=True)
st.write("Telecharger une image au format jpg/png pour que le modèle la classifie par les 10 classes prévu.")


img_file = st.file_uploader("Choisir une image correspondant au sujet CIFAR-10", type = ["jpg", "png", "jpeg"])

if img_file is not None:
    img = Image.open(img_file).convert("RGB")
    st.image(img, caption = "Image téléchargée", use_column_width = True)
    
    img_resized = img.resize((32, 32))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis = 0)

    prediction_model = model.predict(img_array)
    class_prediction = cifar10_class_names[np.argmax(prediction_model)]
    confidence_index = np.max(prediction_model)

    st.markdown(
    f"""
    <div style="text-align: center; font-size: 20px; background-color: #f0f0f0; 
                padding: 15px; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
            <b>Classe prédite :</b> {class_prediction} <br>
            <b>Confiance :</b> {confidence_index * 100:.2f}%
    </div>
    """, unsafe_allow_html=True
            )

    st.subheader("Résultat de la prédiction du modèle")
    st.write(f"**classe prédite :**{class_prediction}")
    st.write(f"**Confiance sur le résultat :**{confidence_index * 100:.2f}%")

    st.subheader("Probabilités par class")
    for i, score in enumerate(prediction_model[0]):
        st.write(f"{cifar10_class_names[i]}:{score * 100:.2f}%")

    fig, ax = plt.subplots(figsize = (6, 4))
    sns.barplot(x = prediction_model[0], y = cifar10_class_names , palette ="Blues", ax = ax)
    ax.set_xlabel("indice de confiance (%)")
    ax.set_title("Distribution des prédictions de classe")
    ax.set_xlim(0, 1)

    st.pyplot(fig)


        