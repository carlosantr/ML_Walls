# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 15:56:24 2024

@author: USUARIO
"""

import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

#%%Prediction Function - Individual
def prediction_ML_walls_individual(IM, T, H_Tcr, Ar_Mean, ALR_G, IA, Sa, Sv,
                                   var_predict=["PFA_max","rDR_max"], model_predict=["RF", "ANN"]):
    """
    This function make ONE prediction of the maximum roof acceleration (PFA_max) and the maximum 
    roof drift ratio (rDR_max) in reinforced concrete wall buildings. The required data to make the 
    predictions is presented below:

    IM: Wall Index
    T: Fundamental period 
    H_Tcr: Stiffness Index
    Ar_Mean: Mean wall aspect ratio
    ALR_G: Average axial load ratio for gravity loads 
    IA: Arias Intensity
    Sa: Spectral acceleration 
    Sv: Spectral velocity
    ___________________________________________________________________
    *The next variables are optional according to the user preference*
    var_predict: The variables to predict
        PFA_max:
        rDR_max:      
    model_predict: The Machine Learning models to do the predictions
        RF: Random Forests
        ANN: Artificial Neural Networks    
    """
    df_prediction = pd.DataFrame(index=model_predict, columns=var_predict)#Dataframe to save predictions
    for var in var_predict:
        for model in model_predict:
            #Import Scalers
            path_scalerX = f"Models/Scalers/ScalerX_{var}_{model}.pkl"
            path_scalerY = f"Models/Scalers/ScalerY_{var}_{model}.pkl"
            scalerX = joblib.load(path_scalerX)
            scalerY = joblib.load(path_scalerY)
            #Import models
            if model == "ANN":
                path_model = f"Models/{var}/{var}_{model}.h5"
                regressor = load_model(path_model, custom_objects={'mse': MeanSquaredError()})
            elif model == "RF":
                path_model = f"Models/{var}/{var}_{model}.pkl"
                regressor = joblib.load(path_model)
            #Creating dataframe of predictors (and scaling the data)
            X = pd.DataFrame({'IM-Arq':[IM],
                              'T-Arq':[T],
                              'H/Tcr-Arq':[H_Tcr],
                              'Ar_Mean':[Ar_Mean],
                              'ALR-G (%)':[ALR_G],
                              'IA':[IA],
                              'Sa':[Sa],
                              'Sv':[Sv]})
            X_scaled = scalerX.transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            #Prediction
            prediction_scaled = regressor.predict(X_scaled)
            prediction = scalerY.inverse_transform(prediction_scaled.reshape(-1, 1)).ravel()
            #Saving the prediction
            df_prediction.loc[model, var] = prediction
            
    return df_prediction

#%%Prediction function - Dataframe
def prediction_ML_walls_multiple(X,
                                 var_predict=["PFA_max","rDR_max"], model_predict=["RF", "ANN"]):
    """
    This function make MULTIPLE predictions of the maximum roof acceleration (PFA_max) and the maximum 
    roof drift ratio (rDR_max) in reinforced concrete wall buildings. The required data to make the 
    predictions is presented below:
    
    X is a dataframe with the next columns (each row is a prediction):
        IM-Arq: Wall Index
        T-Arq: Fundamental period 
        H/Tcr-Arq: Stiffness Index
        Ar_Mean: Mean wall aspect ratio
        ALR-G (%): Average axial load ratio for gravity loads 
        IA: Arias Intensity
        Sa: Spectral acceleration 
        Sv: Spectral velocity
        
        ***************************************
        The columns must be in the same order and must have the same name
        i.e., X.columns = ['IM-Arq', 'T-Arq', 'H/Tcr-Arq', 'Ar_Mean', 'ALR-G (%)', 'IA', 'Sa', 'Sv']
        ***************************************
    ___________________________________________________________________
    *The next variables are optional according to the user preference*
    var_predict: The variables to predict
        PFA_max:
        rDR_max:   
    model_predict: The Machine Learning models to do the predictions
        RF: Random Forests
        ANN: Artificial Neural Networks    
    """
    #Verifying if X have the necessary columns
    X_col = ['IM-Arq', 'T-Arq', 'H/Tcr-Arq', 'Ar_Mean', 'ALR-G (%)', 'IA', 'Sa', 'Sv']
    if list(X.columns) != X_col:
        ValueError(f"The dataframe X does not have the correct columns: {X_col}")
    #Creating prediction columns
    col = [f"{var} - {model}" for var in var_predict for model in model_predict]
    df_prediction = pd.DataFrame(columns=col)
    for var in var_predict:
        for model in model_predict:
            #Import Scalers
            path_scalerX = f"Models/Scalers/ScalerX_{var}_{model}.pkl"
            path_scalerY = f"Models/Scalers/ScalerY_{var}_{model}.pkl"
            scalerX = joblib.load(path_scalerX)
            scalerY = joblib.load(path_scalerY)
            #Import models
            if model == "ANN":
                path_model = f"Models/{var}/{var}_{model}.h5"
                regressor = load_model(path_model, custom_objects={'mse': MeanSquaredError()})
            elif model == "RF":
                path_model = f"Models/{var}/{var}_{model}.pkl"
                regressor = joblib.load(path_model)
            #Scaling the data
            X_scaled = scalerX.transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            #Prediction
            prediction_scaled = regressor.predict(X_scaled)
            prediction = scalerY.inverse_transform(prediction_scaled.reshape(-1, 1)).ravel()
            #Saving the predictions
            df_prediction[f"{var} - {model}"] = prediction
            
    return df_prediction
