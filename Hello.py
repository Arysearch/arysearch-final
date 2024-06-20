import streamlit as st

st.markdown("""
    <style>
        body {text-align: center;}
    </style>
    """, unsafe_allow_html=True)

st.header(':orange[DIABETES DETECTOR]', divider='orange')

col1, col2 = st.columns(2)
with col1:
  a = st.number_input("UMUR : ",min_value=0, value=0)
with col2:
  b = st.number_input("KADAR GULA DARAH (MG/DL) : ",min_value=0, value=0)

col3, col4, col5 = st.columns(3)
with col3:
  pil1 = st.selectbox("JENIS KELAMIN : ",("LAKI LAKI", "PEREMPUAN"),index=none)
with col4:
  pil2 = st.selectbox("PEROKOK : ",("YA", "TIDAK"),index=none)
with col5:  
  pil3 = st.selectbox("RIWAYAT DIABETES KELUARGA : ",("ADA", "TIDAK ADA"),index=none)
c = 0 if pil1 == 'PEREMPUAN' else 1
d = 0 if pil2 == 'TIDAK' else 1
e = 0 if pil3 == 'TIDAK ADA' else 1

col6, col7 = st.columns(2)
with col6:
  f = st.number_input("TINGGI BADAN (M) : ",min_value=1.0, value=1.0)
with col7:
  g = st.number_input("BERAT BADAN (KG) : ",min_value=0, value=0)
h = g / (f ** 2)

option = st.radio('Pilih Metode Input Data :', ['Upload Excel', 'Input Data'])
if option == 'Upload Excel':
    i = j = k = 0.0
    uploaded_file = st.file_uploader('FILE XLSX')
    if uploaded_file is not None:
        # PROCESS CSV
        try:
            # Read the uploaded Excel file into a DataFrame
            df = pd.read_excel(uploaded_file)

            # Normalize 'Frequency' column
            df['Frequency'] = df['Frequency'] / 1000000000

            # Save modified DataFrame to a temporary file
            file_path = "temp_modified_data.xlsx"  # Adjust path if needed
            df.to_excel(file_path, index=False)

            # Read back specific columns
            kolom = ['Frequency', 's11-magnitude (db)']
            df = pd.read_excel(file_path, usecols=kolom)

            # Calculate return loss and related values
            nilaiterkecil = df[df['s11-magnitude (db)'] < 0]['s11-magnitude (db)']
            j = nilaiterkecil.min()
            baris_rl = df[df['s11-magnitude (db)'] == j].index[0] + 1
            i = df.at[baris_rl, 'Frequency']
            list_rl_atas = []  
            list_rl_bawah = []

             # Deret Nilai Batas Atas
            NA = baris_rl
            while NA < len(df):
                nilai = df.iloc[NA]['s11-magnitude (db)']
                if nilai < -10:
                    list_rl_atas.append(nilai)
                else:
                    break
                NA += 1

            # Deret Nilai Batas Bawah
            NB = baris_rl
            while NB >= 0:
                nilai = df.iloc[NB]['s11-magnitude (db)']
                if nilai < -10:
                    list_rl_bawah.append(nilai)
                else:
                    break
                NB -= 1

            #Nilai Return Loss Batas Bawah dan Atas
            nilai_rl_atas = list_rl_atas[-1]
            baris_rl_atas = df[df['s11-magnitude (db)'] == nilai_rl_atas].index[0] + 1
            nilai_rl_bawah = list_rl_bawah[-1]
            baris_rl_bawah = df[df['s11-magnitude (db)'] == nilai_rl_bawah].index[0] + 1
            rentan = nilai_rl_atas - nilai_rl_bawah
            k = abs(rentan*100).round(5)

        except Exception as e:
            st.error("Terjadi kesalahan saat pemrosesan CSV:")
            st.error(e)  

else:
  col8, col9, col10 = st.columns(3)
  with col8:
    i = st.number_input("FREKUENSI (GHz) : ",min_value=0.0, value=0.0)
  with col9:
    j = st.number_input("RETURN LOSS (dB) : ", min_value=-1000.0,max_value=0.0, value=0.0)
  with col10:  
    k = st.number_input("BANDWIDTH (MHz) : ",min_value=0.0, value=0.0)

import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from sklearn.neighbors import KNeighborsClassifier

#Load Data and review content
dataset = pd.read_csv("S1000Data.csv")

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
dataset['Tipe'] = label_encoder.fit_transform(
                                dataset['Tipe'])#Convert input to numpy array
np_data = dataset.to_numpy()

X_data = np_data[:,0:11]
Y_data=np_data[:,11]
X = dataset.drop('Tipe', axis=1)
y = dataset['Tipe']

#Create a scaler model that is fit on the input data.
scalerANN = StandardScaler().fit(X_data)
X_data = scalerANN.transform(X_data)

scalerKNN = StandardScaler()
X_scaled = scalerKNN.fit_transform(X)
# Simpan objek StandardScaler

ANNmodel = keras.models.load_model('ANN_Model.h5')
knn_model = joblib.load('KNN_model.pkl')

input = [[c,a,d,f,g,h,e,b,i,j,k]]
print("Prediction Input :", input)

if st.button("Submit") :
  st.header(':orange[HASIL PERHITUNGAN]', divider='orange')
  col11, col12 = st.columns(2)
  with col11:
    st.latex(r"\text{HASIL PREDIKSI ANN}")
    skalaANN = scalerANN.transform(input)
    prediksiANN = ANNmodel.predict(skalaANN)
    print("PROBABILITAS ANN :" , prediksiANN)
    hasilANN = np.argmax(prediksiANN)
    print("PREDIKSI ANN ", label_encoder.inverse_transform([hasilANN]))
    if hasilANN == 0 :
        print("NON DIABETES")
        st.subheader("`NON DIABETES`")
    else :
        print("DIABETES")
        st.subheader("`DIABETES`")

  with col12:
    st.latex(r"\text{HASIL PREDIKSI KNN}")
    skalaKNN = scalerKNN.transform(input)
    hasilKNN = knn_model.predict(skalaKNN)

    print("Predicted Class:", hasilKNN)
    if hasilKNN == 0 :
        print("NON DIABETES")
        st.subheader("`NON DIABETES`")
    else :
        print("DIABETES")
        st.subheader("`DIABETES`")
