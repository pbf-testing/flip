#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import csv

from math import sin, log, exp
from math import pi

import scipy
import pandas as pd
import pickle
import pingouin as pg

import matplotlib.pyplot as plt

from numpy.random import uniform, seed
from numpy import arange
from numpy import vstack
from numpy import argmax
from numpy import asarray
from numpy.random import normal
from numpy.random import random
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Save a model
def saveModel(filename, path):
    modelAddress = path + filename
    pickle.dump(rf, open(modelAddress , 'wb'))
    
    return None

# Open or load a model
def openModel(filename, path):
    modelAddress = path + filename
    model = pickle.load(open(modelAddress, 'rb'))
    
    return model


# Process data to train a version of fatty liver index plus regression prediction model that integrates into liver fat classification model
def train_fli_plus(dashboard):
    
    dashboard.rename(columns={'pdff_erosion_median_ideal.liver': 'pdff_erosion_median_ideal_liver'}, inplace=True) 
    dashboard['pdff_erosion_median_ideal_liver'] = dashboard.pdff_erosion_median_ideal_liver.astype("float")
    dashboard.drop(dashboard[dashboard['pdff_erosion_median_ideal_liver'] <= 0].index, inplace = True) 
    
    dashboard['ast_alt'] = dashboard['Aspartate_aminotransferase']/dashboard['Alanine_aminotransferase']
    dashboard['ast_plt'] = dashboard['Aspartate_aminotransferase']/dashboard['Platelet_count']
    dashboard['waist_hip'] = dashboard['Waist_circumference']/dashboard['Hip_circumference']
    
    # uncomment this line if 'liver_disease_diagnosis_2' is binary in CSV dataset
    # dashboard['liver_disease_diagnosis_2'] = dashboard['liver_disease_diagnosis_2'].map(
    #                      {1:True ,0:False})
    
    dashboard['Triglycerides'] = 18 * dashboard['Triglycerides']
    dashboard['Urate'] = 0.0113 * dashboard['Urate']
        
        
    features = pd.DataFrame()
    feature_list = ['Urate',
                   'Testosterone',
                   'Body_mass_index_BMI',
                   'Gamma_glutamyltransferase',
                   'Aspartate_aminotransferase',
                   'Alanine_aminotransferase',
                   'age_when_attended_assessment_centre',
                   'ast_alt',
                   'ast_plt',
                   'waist_hip',
                   'liver_disease_diagnosis_2',
                   
                   'Triglycerides', 
                   'Waist_circumference',
                   'pdff_erosion_median_ideal_liver',
                   
                   ]
    
    # print(len(feature_list))
    for i in range(len(feature_list)):
        features[feature_list[i]] = dashboard[feature_list[i]];

    # drop NaNs
    features = features.dropna()
    
    dummy_c = ['liver_disease_diagnosis_2',
               # 'type_2_diabetes_mellitus_diagnosis'
               # 'Amount_of_alcohol_drunk_on_a_typical_drinking_day',
               ]
    
    for i in dummy_c:
        features = features.join(pd.get_dummies(features[i], prefix=i))      
    features.drop(dummy_c, axis=1, inplace=True)

    tri = np.array(features['Triglycerides']).flatten()
    bmi = np.array(features['Body_mass_index_BMI']).flatten()
    ggt = np.array(features['Gamma_glutamyltransferase']).flatten()
    wcc = np.array(features['Waist_circumference']).flatten()
    
    mu = 0.953 * np.log(tri) + 0.139 * bmi + 0.718 * np.log(ggt) + 0.053 * wcc - 15.745 
    num = np.exp(mu)
    den = 1 + np.exp(mu)
    fli_compute = (num/den)*100
    #fli_compute = np.interp(fli_compute, (fli_compute.min(), fli_compute.max()), (0, 100))
    
    features['fatty liver index'] = fli_compute
    # shift to first column
    first_column = features.pop('fatty liver index')
    features.insert(0, 'fatty liver index', first_column)
    
    # remove/drop Triglycerides from features
    features.drop('Triglycerides', axis=1, inplace=True)
    
    # remove/drop Waist_circumference from features
    features.drop('Waist_circumference', axis=1, inplace=True)
    
    # check feature_list
    feature_list00 = list(features.columns)
    
    # gather ground-truth and interpolate to map ranges to fli_compute
    pdff = features['pdff_erosion_median_ideal_liver']
    pdff = np.array(pdff).flatten()
    
    y05 = np.where((pdff <= 5))
    y0510 = np.where((pdff > 5) & (pdff <= 10))
    y10more = np.where((pdff > 10))
        
    check01 = 29.0+(0.99999999999999)
    check02 = 59.0+(0.99999999999999)
                         
    pdff[y05] = np.interp(pdff[y05], (pdff[y05].min(), pdff[y05].max()), (0, check01))
    pdff[y0510] = np.interp(pdff[y0510], (pdff[y0510].min(), pdff[y0510].max()), (30, check02))
    pdff[y10more] = np.interp(pdff[y10more], (pdff[y10more].min(), pdff[y10more].max()), (60, 100))
        
    # remove/drop ground-truth from features (pdff_erosion_median_ideal_liver)
    features.drop('pdff_erosion_median_ideal_liver', axis=1, inplace=True)
    
    # check feature_list (again)
    feature_list01 = list(features.columns)
     
    
    """
    # Check plot
    ax = plt.axes()
    plt.scatter(pdff,features['fatty liver index'],color='blue',)
    ax.set_facecolor("#DCDCDC")
    ax.tick_params(axis='both', labelsize=16)
    
    x0 = pdff.copy()
    y0 = np.array(features['fatty liver index']).flatten()
    m, b = np.polyfit(x0, y0, 1)
    plt.plot(x0, m*x0 + b, color='magenta', lw=3, scalex=False, scaley=False)
        
    # xpoints = ypoints = plt.xlim()
    # plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=3, scalex=False, scaley=False)
    plt.grid(b=True, which='major', color='#666666', linestyle='-')

    correlation_matrix = np.corrcoef(pdff, np.array(features['fatty liver index']).flatten())
    correlation_xy = correlation_matrix[0,1]
    r_squared = round(correlation_xy**2,3)
             
    plt.xlabel('PDFF (%)', fontsize=16)
    plt.ylabel('FLI', fontsize=16) 
    plt.title ('FLI versus Mapped PDFF(%) - ' + '$R^{2}$: ' + str(r_squared), fontsize=16)
   
    plt.savefig(path + "FLI_versus_mapped_PDFF.png", bbox_inches='tight')
    plt.show(); plt.close()
    """
    
    # take the log of the ground-truth (pdff)
    # pdff_log = np.log(pdff)
    minsplit = 0.15 #0.1 #0.2 #0.25
    minval = 0.0000000001
    pdff_log = np.log(pdff.clip(min=minval))
    
    # convert features to numpy
    features = np.array(features)     
    
    # split data to obtain training and testing features and labels (ground-truth)
    train_features, test_features, train_labels, test_labels = train_test_split(features, pdff_log, test_size = minsplit, random_state = 42)
    
    # define model
    gbr_params = {'n_estimators': 100,
              'max_depth': 100,
              'min_samples_split': 5,
              'learning_rate': 0.1,
              'loss': 'ls',
              'subsample':0.75}
            
    model = GradientBoostingRegressor(**gbr_params)
    
    # train model
    model.fit(train_features, train_labels)
    
    """
    # evaluate model
    predictions = model.predict(test_features)
    test_labels = np.exp(test_labels)
    predictions = np.exp(predictions)
    
    adj_r2 = 1 - (1-r2_score(test_labels, predictions)) * (len(test_labels)-1)/(len(test_labels)-test_features.shape[1]-1)
    coeffd = r2_score(test_labels, predictions)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
            
    print('Training Model Performance')
    print('Coefficient of determination (R^2) = {:0.4f}.'.format(coeffd))
    print('Adjusted coefficient of determination (R^2) = {:0.4f}.'.format(adj_r2))
    print('Mean Bias Error: {:0.4f}, std = {:0.4f}'.format(np.mean(errors), np.std(errors)))
    print('MAPE: {:0.4f}%.'.format(mape))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    ax = plt.axes()
    plt.scatter(test_labels,predictions,color='blue')
    ax.set_facecolor("#DCDCDC")
    ax.tick_params(axis='both', labelsize=16)
        
    xpoints = ypoints = plt.xlim()
    plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=3, scalex=False, scaley=False)
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
   
    plt.xlabel('Ground-truth', fontsize=16); 
    plt.ylabel('Prediction', fontsize=16); 
    plt.title ('FLI Plus for liver fat classification - ' + '$R^{2}$: ' + str(round(adj_r2,3)), fontsize=16)
    
    plt.savefig(path + "FLIPlus_versus_Mapped_PDFF.png", bbox_inches='tight')
    plt.show(); plt.close()
    """
    
    # save model
    modelAddress = path + 'fatty_liver_index_plus_ori.sav'
    pickle.dump(model, open(modelAddress , 'wb'))
   
    return None

def apply_fli_plus_csv(dashboard):
    

    # define dataframe to contain variables (features) of interest
    features = pd.DataFrame()
   
    # define initial variables to extract from CSV file (if given in CSV file)
    # add additional variables (if available in CSV file)
    feature_list = [ 'individual_id','weight','height',
                    'age_when_attended_assessment_centre',
                    'Sex','Waist_circumference','Hip_circumference',
                    'Body_mass_index_BMI',
                    'Gamma_glutamyltransferase',
                    'Triglycerides',
                    'Urate',
                    'Platelet_count',
                    'Testosterone',
                    'Aspartate_aminotransferase',
                    'Alanine_aminotransferase',
                    'liver_disease_diagnosis_2',
             ]
    
    # print(len(feature_list))
    for i in range(len(feature_list)):
        features[feature_list[i]] = dashboard[feature_list[i]];
  
    # make sure that blood biochemical units are in accordance with FLI algorithm
    features['Triglycerides_ori'] = features['Triglycerides']
    features['Triglycerides'] = 18 * features['Triglycerides']
    features['Urate_ori'] = features['Urate']
    features['Urate'] = 0.0113 * features['Urate']
    
    features['liver_ori'] = features['liver_disease_diagnosis_2']
    # features['liver_disease_diagnosis_2'] = features['liver_disease_diagnosis_2'].map({1:True ,0:False})

    features['ast-alt'] = features['Aspartate_aminotransferase']/features['Alanine_aminotransferase']
    features['ast-plt'] = features['Aspartate_aminotransferase']/features['Platelet_count']
    features['waist-hip'] = features['Waist_circumference']/features['Hip_circumference']
    
    features = features.dropna()
    
    # make sure that blood biochemical units are in accordance with FLI algorithm
    tri = np.array(features['Triglycerides']).flatten()
    bmi = np.array(features['Body_mass_index_BMI']).flatten()
    ggt = np.array(features['Gamma_glutamyltransferase']).flatten()
    wcc = np.array(features['Waist_circumference']).flatten()
    mu = 0.953 * np.log(tri) + 0.139 * bmi + 0.718 * np.log(ggt) + 0.053 * wcc - 15.745 
    num = np.exp(mu); den = 1 + np.exp(mu); fli_compute = ((num/den)*100)
    features['fatty-liver-index'] = fli_compute
    
    # construct dataframe to apply fatty livwer plus model
    # extract data from initial dataframe, 'features' into checkFLI
    checkFLI = pd.concat([
        features['fatty-liver-index'], 
        features['Urate'],
        features['Testosterone'] ,
        features['Body_mass_index_BMI'],
        features['Gamma_glutamyltransferase'],
        features['Aspartate_aminotransferase'],
        features['Alanine_aminotransferase'],
        features['age_when_attended_assessment_centre'],
        features['ast-alt'],
        features['ast-plt'],
        features['waist-hip'],
        features['liver_disease_diagnosis_2']
        ], axis=1)
    
    checkFLI_c = ['liver_disease_diagnosis_2', 
                  # 'type_2_diabetes_2'
                 ]
    
    for i in checkFLI_c:
        checkFLI = checkFLI.join(pd.get_dummies(checkFLI[i], prefix=i))
    checkFLI.drop(checkFLI_c, axis=1, inplace=True)
    # checkFLI_list = list(checkFLI.columns)
    
    # create a copy of checkFLI to add and trasfer data to a new CSV file
    checkFLI2 = checkFLI.copy()
    
    # convert to numpy array and apply data to 'fli_plus_model'.
    checkFLI = np.array(checkFLI)
    fli_plus_model = pickle.load(open(path0+'fatty_liver_index_plus_ori.sav', 'rb'))
    fli_plus_values = fli_plus_model.predict(checkFLI)
    fli_plus_values = np.exp(fli_plus_values)
    
    # drop columns prior to adding columns to new CSV file
    checkFLI2 = checkFLI2.drop('Urate', axis = 1)
    checkFLI2 = checkFLI2.drop('liver_disease_diagnosis_2_False', axis = 1)
    checkFLI2 = checkFLI2.drop('liver_disease_diagnosis_2_True', axis = 1)
    
    # add desired variables (if given in original CSV file) to dataframe
    checkFLI2['individual_id'] = features['individual_id']
    checkFLI2['Sex'] = features['Sex']
    checkFLI2['weight'] = features['weight']
    checkFLI2['height'] = features['height']
    checkFLI2['Waist_circumference'] = features['Waist_circumference']
    checkFLI2['Hip_circumference'] = features['Hip_circumference']
    checkFLI2['Platelet_count'] = features['Platelet_count']
    checkFLI2['Triglycerides_ori'] = features['Triglycerides_ori']
    checkFLI2['Urate_ori'] = features['Urate_ori']
    checkFLI2['liver_ori'] = features['liver_ori']
    
    checkFLI2.rename(columns={'Triglycerides_ori': 'Triglycerides'}, inplace=True) 
    checkFLI2.rename(columns={'Urate_ori': 'Urate'}, inplace=True) 
    checkFLI2.rename(columns={'liver_ori': 'liver_disease_diagnosis_2'}, inplace=True) 

    # add predicted fatty liver plus values
    checkFLI2['fatty-liver-index-plus'] = fli_plus_values
    
    # create header list for new CSV file
    column_names = ["individual_id",
                    "Sex",
                    "age_when_attended_assessment_centre", 
                    "weight", 
                    "height", 
                    "Waist_circumference", 
                    "Hip_circumference",
                    "waist-hip",
                    "Body_mass_index_BMI",
                    "Testosterone",
                    "Urate",
                    "Triglycerides",
                    "Gamma_glutamyltransferase",
                    "Aspartate_aminotransferase",
                    "Alanine_aminotransferase",
                    "Platelet_count",
                    "ast-alt",
                    "ast-plt",
                    "liver_disease_diagnosis_2",
                    "fatty-liver-index-plus"
                    ]
    
    checkFLI2 = checkFLI2.reindex(columns=column_names)
    
    
    checkFLI_list = list(checkFLI2.columns)
    checkFLI2.to_csv(path + 'fatty_liver_index_plus_results' + '.csv', 
                         index=False, columns=checkFLI_list, header=checkFLI_list, na_rep='NA')
    
        
    return None

def apply_fli_plus_single(age,bmi,waist_cm,hip_cm,
                          ggt,trig,urate,platelet_count,ast,alt,
                          testosterone,
                          liver_disease):

    
    # in case 'liver_disease' is given as a string variable
    if liver_disease == 'True' or liver_disease == 'true':
        liver_disease = True
    elif liver_disease == 'False' or liver_disease == 'false':
        liver_disease = False

    # make sure all continous variables are float 
    age = float(age)
    waist_cm = float (waist_cm)
    hip_cm = float(hip_cm)
    bmi = float(bmi)
    ggt = float(ggt)
    trig = float(trig)
    urate = float(urate)
    testosterone = float(testosterone)
    platelet_count = float(platelet_count)

    # convert to mg/dL
    # based on original values
    # in training dataset
    trig = 18 * trig
    urate = 0.0113 * urate

    ast_alt = ast/alt
    ast_plt = ast/platelet_count
    waist_hip = waist_cm/hip_cm

    mu = 0.953 * np.log(trig) + 0.139 * bmi + 0.718 * np.log(ggt) + 0.053 * waist_cm - 15.745
    num = np.exp(mu)
    den = 1 + np.exp(mu)
    fli = (num/den)*100
   
    list_liver_disease = [False,True]
    list_liver_disease.remove(liver_disease)

    feature_full_list = ['fatty-liver-index',
                         'Urate',
                         'Testosterone',
                         'Body_mass_index_BMI',
                         'Gamma_glutamyltransferase',
                         'Aspartate_aminotransferase',
                         'Alanine_aminotransferase',
                         'age_when_attended_assessment_centre',
                         'ast-alt',
                         'ast-plt',
                         'waist-hip',
                         'liver_disease_diagnosis_2']

    test_features=[[fli,urate,testosterone,bmi,
                    ggt,ast,alt,age,
                    ast_alt,ast_plt,waist_hip,
                    liver_disease]]

    features = pd.DataFrame(test_features, columns = feature_full_list)

    data = []

    for l in range(len(list_liver_disease)):

        values=[np.nan,np.nan,np.nan,np.nan,
                np.nan,np.nan,np.nan,np.nan,
                np.nan,np.nan,np.nan,
                list_liver_disease[l]]

    zipped = zip(feature_full_list, values)
    a_dictionary = dict(zipped)
    data.append(a_dictionary)
    vvt = features.append(data, True)

    vvt_c = ['liver_disease_diagnosis_2']
    for i in vvt_c:
        vvt = vvt.join(pd.get_dummies(vvt[i], prefix=i))
    vvt.drop(vvt_c, axis=1, inplace=True)
    vvt = vvt.dropna()

    fli_plus_model = pickle.load(open(path0+'fatty_liver_index_plus_ori.sav', 'rb'))
    fli_plus_value = fli_plus_model.predict(vvt)
    fli_plus_value = np.exp(fli_plus_value)
    fli_plus_value = fli_plus_value[0]

    return fli_plus_value
    
    
arr = [1]
# while(not sleep(5)):
for xi in range(len(arr)):

    path0 = '/Volumes/maxone/flip/'
    mod = 'flip'; date = '28_01';
    path = '/Volumes/maxone/flip/' + mod + '_model' + '_' + date + '/'
    
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory) 

    # Load multiple subject data from CSV file 
    # into dataframe (dashboard)
    filename = path0 + 'fields_and_idps_29_09_21.csv'
    dashboard = pd.read_csv(filename)
    dashboard.drop_duplicates(subset=None, keep="first", inplace=True)

    """
    # 1. Train fatty liver index plus model
    # using multiple subject data (dashboard) 
    # from a CSV file  and save model to file - START
    
    train_fli_plus(dashboard)
    
    # 1. Train fatty liver index plus model - END  
    """

    # 2. Compute multiple fatty liver index plus scores by 
    # accessing multiple subject data (dashboard) from a CSV file,
    # and then save results to a CSV file - START
    
    apply_fli_plus_csv(dashboard)
    
    # 2. Compute multiple fatty liver index plus scores - END  
    
    
    # 3. Single test (female) subject example - START
    weight_kg = 58.9 # kg
    height_cm = 164 # cm
    bmi = weight_kg/((height_cm/100)*(height_cm/100)) # kg/m^2
    age = 73 # years
    waist_cm = 78 # cm
    hip_cm = 95; # cm
    ggt =  53.1 # U/L
    trig = 1.544 # mmol/L
    urate = 265.3 # umol/L
    platelet_count = 343 # 10^9/L
    ast = 26.5 # UL
    alt = 19.27 #  U/L
    testosterone = 0.863 # nmol/L
    liver_disease = False
    
    # 3i. Compute fatty liver index plus for single test subject 
    fli_plus_value = apply_fli_plus_single(age,bmi,waist_cm,hip_cm,
                                           ggt,trig,urate,platelet_count,ast,alt,
                                           testosterone,
                                           liver_disease)
    
    print('Fatty liver index plus is: ' + str(fli_plus_value))
    
    # 3. Single test (female) subject example - END