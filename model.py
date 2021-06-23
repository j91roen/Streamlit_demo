import numpy as np
import pandas as pd

def calculate_diagnosis(data, RFE, symptom_duration):
    data_array = data.to_numpy()

    weights = np.random.randn(7,10)
    bias = np.random.randn(10) + 20*np.array([symptom_duration,0,0,0,0,0,0,0,0,0])
    if RFE == 'B01':
        bias[1]= bias[1]+1000
    output = np.matmul(data_array[:,1],weights) + bias
    output = output - np.min(output)
    output = output/np.sum(output)
    id_max = np.argmax(output)
    output_df = pd.DataFrame(
        np.array([['Pneumonia', 'Asthma', 'COPD','Diabetes', 'Bronchitis', 'Cardiac arrest','Embolism','Allergy','COVID','Influenza'],output]).T,
        columns=['diagnoses', 'probabilities'])

    output_df = output_df.sort_values(by=['probabilities'])

    prior_probabilities = np.random.rand(10)
    prior_probabilities = prior_probabilities / np.sum(prior_probabilities)

    features_df = pd.DataFrame(
        np.array([data['variables'],weights[:,id_max]]).T,
        columns=['features', 'weights'])

    return output_df, prior_probabilities, features_df