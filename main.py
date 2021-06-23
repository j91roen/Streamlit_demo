import streamlit as st
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from model import calculate_diagnosis

### Settings
plt.style.use('dark_background')
np.random.seed(1)
symptom_duration = 1
RFE = 'A01'

### Display title and description
st.title('AI for Health demo')
st.write("This application predicts the top 10 most probable diagnoses, based on a combination of a patients medical history and most recent symptoms.")

# ### Create dataframe
# chart_data = pd.DataFrame(
#      np.random.rand(10),
#      columns=['test'])
#
# st.write(chart_data)
#
# ### Plot data
# st.line_chart(chart_data)

### Add selectbox
RFEs = ['A01', 'A02', 'B01', 'C01']
RFE = st.sidebar.selectbox(
    'Please select an RFE.',
    RFEs)

### Add slider
symptom_duration = st.sidebar.slider('Duration of symptoms (days):', 0, 30, 10)


### Enable file uploading
st.write("Please upload a CSV file.")
uploaded_file = st.file_uploader("Choose a file")

### Display dataframe
if uploaded_file is not None:

    dataframe = pd.read_csv(uploaded_file)

    #st.write(dataframe)

    ### Toggle dataframe
    if st.sidebar.checkbox('Show patient data'):
        st.sidebar.write(dataframe)

    #### Progress bar
    # latest_iteration = st.empty()
    # bar = st.progress(0)
    #
    # for i in range(100):
    #   latest_iteration.text(f'Iteration {i+1}')
    #   bar.progress(i + 1)
    #   time.sleep(0.01)
    # st.write("Analysis successful")

    ### Calculate diagnosis
    output_df, prior_probabilities, features_df = calculate_diagnosis(dataframe, RFE, symptom_duration)

    ### Display diagnosis

    fig1 = plt.figure()
    plt.barh(output_df['diagnoses'], width = output_df['probabilities'],height=0.5)
    fig1.patch.set_alpha(0.0)
    plt.title('Top 10 most likely diagnoses')
    plt.xlabel('Probability')

    ### Add prior probabilities
    if st.sidebar.checkbox('Show prior probabilities'):
        for ii in range(10):
            plt.plot([prior_probabilities[ii],prior_probabilities[ii]],[output_df.index[ii]-0.25,output_df.index[ii]+0.25],color='C1',linewidth=5)

    plt.xlim(0,0.25)
    st.pyplot(fig1)

    ### Display weights
    fig2 = plt.figure()
    plt.barh(features_df['features'], width = features_df['weights'],height=0.5)
    plt.title('Feature weights')
    plt.xlabel('Weight')
    fig2.patch.set_alpha(0.0)
    plt.xlim(-3,3)

    st.pyplot(fig2)
