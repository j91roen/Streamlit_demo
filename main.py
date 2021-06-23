import streamlit as st
import numpy as np
import pandas as pd
import time
import matplotlib
import matplotlib.pyplot as plt
import altair as alt


plt.style.use('dark_background')
np.random.seed(1)
symptom_duration = 1
RFE = 'A01'

def calculate_diagnosis(data, RFE, symptom_duration):
    data_array = data.to_numpy()

    weights = np.random.randn(7,10)
    bias = np.random.randn(10) + 20*np.array([symptom_duration,0,0,0,0,0,0,0,0,0])
    output = np.matmul(data_array[:,1],weights) + bias
    output = output - np.min(output)
    output = output/np.sum(output)
    output_df = pd.DataFrame(
        np.array([['Pneumonia', 'Asthma', 'COPD','Diabetes', 'Bronchitis', 'Cardiac arrest','Embolism','Allergy','COVID','Influenza'],output]).T,
        columns=['diagnoses', 'probabilities'])
    if st.checkbox('Sort diagnosis'):
        output_df = output_df.sort_values(by=['probabilities'])

    prior_probabilities = np.random.rand(10)
    prior_probabilities = prior_probabilities / np.sum(prior_probabilities)

    return output_df, prior_probabilities



### Display title and description
st.title('AI for Health demo')
st.write("This application predicts the most probable diagnosis based on a patients information.")

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

st.write('You selected: ', RFE)

### Add slider
symptom_duration = st.sidebar.slider('Duration of symptoms (days):', 0, 30, 10)


### Enable file uploading
st.write("Please upload a CSV file.")
uploaded_file = st.file_uploader("Choose a file")

### Display dataframe
if uploaded_file is not None:

    dataframe = pd.read_csv(uploaded_file)

    st.write(dataframe)

    ### Toggle dataframe
    if st.checkbox('Show dataframe'):
        st.sidebar.write(dataframe)

    #### Progress bar
    # latest_iteration = st.empty()
    # bar = st.progress(0)
    #
    # for i in range(100):
    #   latest_iteration.text(f'Iteration {i+1}')
    #   bar.progress(i + 1)
    #   time.sleep(0.01)

    ### Calculate diagnosis
    output_df, prior_probabilities = calculate_diagnosis(dataframe, RFE, symptom_duration)
    st.write("Analysis successful")

    ### Display diagnosis

    fig1 = plt.figure()
    plt.barh(output_df['diagnoses'], width = output_df['probabilities'],height=0.5)
    fig1.patch.set_alpha(0.0)

    ### Add prior probabilities
    if st.checkbox('Show prior probabilities'):
        for ii in range(10):
            plt.plot([prior_probabilities[ii],prior_probabilities[ii]],[output_df.index[ii]-0.25,output_df.index[ii]+0.25],color='C1',linewidth=5)

    plt.xlim(0,0.25)
    st.pyplot(fig1)


