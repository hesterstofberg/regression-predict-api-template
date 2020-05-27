"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.

    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    
    dftest = feature_vector_df.copy()
    
    dftest = dftest.drop(['Precipitation in millimeters'],axis=1)
    
    dftest.columns = [col.replace(" ", "_") for col in dftest.columns]
    
    dftest = dftest.drop(['Order_No','User_Id','Vehicle_Type','Platform_Type',
              'Personal_or_Business','Placement_-_Weekday_(Mo_=_1)',
              'Placement_-_Time','Confirmation_-_Day_of_Month','Confirmation_-_Weekday_(Mo_=_1)',
              'Confirmation_-_Time','Arrival_at_Pickup_-_Weekday_(Mo_=_1)', 'Temperature',
            'Pickup_-_Time','Pickup_-_Weekday_(Mo_=_1)',
              'Rider_Id', 'Arrival_at_Pickup_-_Day_of_Month', 
            'Pickup_-_Day_of_Month',
                       'Placement_-_Day_of_Month'],axis=1)
    
    dftest_Arrival_at_Pickup_Time_hr = []

    #Arrival_at_Pickup_Time_min = []
    #Arrival_at_Pickup_Time_sec = []

    for t in range(len(dftest)):
    
        # Split the timestamp at ':'
        time_spl= dftest['Arrival_at_Pickup_-_Time'][t].split(':')
    
        # The 'seconds' part still has the 'AM and PM'. eg: '14 AM'
        time_sec= time_spl[-1].split()
    
        #Arrival_at_Pickup_Time_min.append(int(time_spl[1]))
        #Arrival_at_Pickup_Time_sec.append(int(time_sec[0]))
    
        # Check if its 'AM' and if the first element is 12. If it is 12
        # then append 0 which corresponds to 12 AM.
        if (time_sec[1] == 'AM') and (int(time_spl[0]) == 12):
            dftest_Arrival_at_Pickup_Time_hr.append(0)
    
        # Check if its 'PM' and if the first element is 12. If it is 12
        # then append 12 which corresponds to 12 PM.
        elif (time_sec[1] == 'PM') and (int(time_spl[0]) == 12):
            dftest_Arrival_at_Pickup_Time_hr.append(12)
    
        # Check if its 'AM'. If it is, then leave it as is, because it corresponds to 
        # AM hours.
        elif (time_sec[1] == 'AM'):
            dftest_Arrival_at_Pickup_Time_hr.append(int(time_spl[0]))
    
        # Else add 12 to the hour to convert it to 24-hour format
        else:
            dftest_Arrival_at_Pickup_Time_hr.append(int(time_spl[0]) + 12)

    # Insert the converted hour in place of the timestamp
    dftest.insert(0, 'Arrival_at_Pickup_-_Time_Hours', dftest_Arrival_at_Pickup_Time_hr, True)
    dftest.drop('Arrival_at_Pickup_-_Time', axis = 1, inplace = True)
    
    Xtest = dftest
    
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    
    Xtest_scaled = scaler.fit_transform(Xtest)
    
    Xtest_standardise = pd.DataFrame(Xtest_scaled,columns=Xtest.columns)
    
    predict_vector = Xtest_standardise.copy()
    
    # ------------------------------------------------------------------------

    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))

def make_prediction(data, model):
    """Prepare request data for model prediciton.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standerdisation.
    return prediction[0].tolist()
