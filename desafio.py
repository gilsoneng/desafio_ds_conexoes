# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 11:48:03 2021

@author: altran
"""

import pandas as pd
import numpy as np
import sys

import tensorflow as tf
import math
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import sklearn.metrics
from sklearn.preprocessing import LabelEncoder
import datetime
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import pickle

def inverse_dict(my_dict):
    """
    the func get a dictinary and reverse it, the keys become values and the values become keys.
    :param my_dict: the dictinary that need to be reversed.
    :return: a VERY pretty dictionary.
    """
    result_dict = {}
    for key, value in my_dict.items():
        if not value in result_dict.keys():
            result_dict[value] = ''
        result_dict[value]=(key)
    return result_dict

def pd_read_csv(delimiter, folder_raw, filename):
    df = pd.read_csv(filepath_or_buffer = f'{folder_raw}\\{filename}'
                         , delimiter = delimiter)
    return df

def transform_raw_clean(filename
                        , transform
                        , folder_raw = 'RAW'
                        , folder_clean = 'CLEAN'
                        , delimiter = ';'
                        , number_columns = 'all'):
    
    df_raw = pd_read_csv(delimiter = delimiter, folder_raw = folder_raw, filename = filename)
    
    if transform == 'transform_float_column':
        
        df_raw_train = df_raw[df_raw[number_columns].notnull()]
        
        df_raw_previsao = df_raw[df_raw[number_columns].isnull()]
        
        df_raw_train[number_columns] = df_raw[number_columns].astype(float)
        
        df_raw_train.to_csv(f'{folder_clean}\\{filename}_train.csv',sep = delimiter, index = False)
        
        df_raw_previsao.to_csv(f'{folder_clean}\\{filename}_previsao.csv',sep = delimiter, index = False)
        
        return df_raw_train, df_raw_previsao
    
    elif transform == 'get_median_float_column':
        
        number_columns = df_raw.columns if number_columns == 'all' else number_columns
        
        for column in number_columns:
            print(column)
            try:
                median = df_raw[column].median()
                df_raw['column_null'] = df_raw[column].isnull()
                df_raw[column] = df_raw.apply(lambda x: median if x['column_null'] else x[column], axis = 1)
                df_raw = df_raw.drop(columns=['column_null'], axis =1)
            except TypeError:
                mode = df_raw[column].mode()[0]
                df_raw['column_null'] = df_raw[column].isnull()
                df_raw[column] = df_raw.apply(lambda x: mode if x['column_null'] else x[column], axis = 1)
                df_raw = df_raw.drop(columns=['column_null'], axis =1)
        
        df_raw.to_csv(f'{folder_clean}\\{filename}.csv',sep = delimiter, index = False)
    
        return df_raw

def prepare_training_database():
    filename = 'conexoes_espec.csv'
    number_columns = 'prob_V1_V2'
    df_clean_train_conexoes, df_clean_previsao_conexoes = transform_raw_clean(filename
                                        , transform = 'transform_float_column'
                                        , folder_raw = 'RAW'
                                        , folder_clean = 'CLEAN'
                                        , delimiter = ';'
                                        , number_columns = number_columns)
    filename = 'individuos_espec.csv'
    df_transform_individuos = transform_raw_clean(filename
                                        , transform = 'get_median_float_column'
                                        , folder_raw = 'RAW'
                                        , folder_clean = 'CLEAN'
                                        , delimiter = ';'
                                        , number_columns = 'all')
    
    df_transform_individuos_transmissor = df_transform_individuos
    
    for column in df_transform_individuos_transmissor.columns:
        renamed_column = f'{column}_transmissor'.replace(' ', '_')
        df_transform_individuos_transmissor = df_transform_individuos_transmissor.rename(columns={f'{column}': f'{renamed_column}'})
    
    df_transform_individuos_receptor = df_transform_individuos
    
    for column in df_transform_individuos_receptor.columns:
        renamed_column = f'{column}_receptor'.replace(' ', '_')
        df_transform_individuos_receptor = df_transform_individuos_receptor.rename(columns={f'{column}': f'{renamed_column}'})
    
    df_clean_train_conexoes_merged = df_clean_train_conexoes.merge(df_transform_individuos_transmissor, how = 'left', left_on = 'V1', right_on = 'name_transmissor')
    
    df_clean_train_conexoes_merged = df_clean_train_conexoes_merged.merge(df_transform_individuos_receptor, how = 'left', left_on = 'V1', right_on = 'name_receptor')
    
    df_clean_previsao_conexoes_merged = df_clean_previsao_conexoes.merge(df_transform_individuos_transmissor, how = 'left', left_on = 'V1', right_on = 'name_transmissor')
    
    df_clean_previsao_conexoes_merged = df_clean_previsao_conexoes_merged.merge(df_transform_individuos_receptor, how = 'left', left_on = 'V1', right_on = 'name_receptor')
    
    return df_clean_train_conexoes_merged, df_clean_previsao_conexoes_merged

def feature_engineer(df, feature_columns, target_train, cat_columns = ''):
        
    df_cat = pd.DataFrame([])
    
    cat_columns = df.columns if cat_columns == '' else cat_columns
    
    for col_name in cat_columns:
        if(df[col_name].dtype == 'object'):
            df_cat_tmp = pd.DataFrame(df[col_name].unique()).reset_index().rename(columns = {'index': 'code', 0: 'category'})
            dict_tmp = pd.DataFrame(df[col_name].unique()).to_dict()[0]
            inv_dict_tmp = inverse_dict(dict_tmp)
            df_cat_tmp['name_map'] = f'{col_name}'
            df_cat = df_cat.append(df_cat_tmp, ignore_index=True)
            df[f'{col_name}_code'] = df[col_name].apply(lambda x: inv_dict_tmp[x])
            feature_columns.remove(col_name)
            feature_columns.append(f'{col_name}_code')
            
    
    df_train, df_test = train_test_split(df, test_size=0.3)
    
    return df_train, df_test, feature_columns, df_cat

def model_compile(input_numbers, layers_numbers = 6, out_numbers = 1, activation = 'relu', neuron_units = 50):
    model = Sequential()
    model.add(Dense(units = 2 * input_numbers, activation = 'linear', input_dim=input_numbers))
    for add in range(layers_numbers):
        model.add(Dense(units = neuron_units, activation = activation))
    model.add(Dense(units = out_numbers, activation = 'linear'))
    model.compile(loss='mse', optimizer="adam")
    return model

if __name__ == "__main__":
    
    v_modelo = str(datetime.datetime.now()).replace('-','_').replace(' ','_').replace(':','_').replace('.','_')
    
    df, df_previsao = prepare_training_database()
    
    
    feature_columns = ['grau', 'proximidade',
       'idade_transmissor', 'estado_civil_transmissor',
       'qt_filhos_transmissor', 'estuda_transmissor', 'trabalha_transmissor',
       'pratica_esportes_transmissor', 'transporte_mais_utilizado_transmissor',
       'IMC_transmissor', 'idade_receptor',
       'estado_civil_receptor', 'qt_filhos_receptor', 'estuda_receptor',
       'trabalha_receptor', 'pratica_esportes_receptor',
       'transporte_mais_utilizado_receptor', 'IMC_receptor']
    
    target_train = ['prob_V1_V2']
    
    cat_columns = ['grau','proximidade','estado_civil_transmissor','transporte_mais_utilizado_transmissor','estado_civil_receptor','transporte_mais_utilizado_receptor']
    
    df_train, df_test, feature_columns, df_cat = feature_engineer(df, feature_columns, target_train, cat_columns)
    
    df_train.to_csv(f'EVALUATED\\base_treinamento_{v_modelo}.csv',sep = ';', index = False)
    
    df_test.to_csv(f'EVALUATED\\base_teste_{v_modelo}.csv',sep = ';', index = False)
    
    input_numbers = len(feature_columns)
    
    #Codificação de variaveis categóricas com base no treinamento
    
    for cat in df_cat['name_map'].unique():
        dict_tmp = df_cat[df_cat['name_map']==cat][['category', 'code']].reset_index().to_dict()['category']
        inv_dict_tmp = inverse_dict(dict_tmp)
        df_previsao[f'{cat}_code'] = df_previsao[cat].apply(lambda x: inv_dict_tmp[x])
        
    #-----------deep_learning-----------
    
    test_type = 'deep_learning'
    
    model =  model_compile(input_numbers, layers_numbers = 6, out_numbers = 1, activation = 'relu', neuron_units = 50)
    
    model.fit(df_train[feature_columns], df_train[target_train],
          epochs=15, verbose=1, use_multiprocessing=True)
    
    model.save(f"MODELS\\model_{test_type}_{v_modelo}")
    
    Y_Predict = model.predict(df_test[feature_columns])

    df_test['Y_Predict'] = Y_Predict.astype(float)
    
    mse = mean_squared_error(df_test[target_train], df_test['Y_Predict'])

    r2_score = sklearn.metrics.r2_score(df_test[target_train], df_test['Y_Predict'])
    
    print(f'Test Type: {test_type}, Mean Square Error: {mse.round(2)}, R²: {r2_score.round(2)}')
    
    #predição do modelo
        
    df_previsao['Y_Predict'] = model.predict(df_previsao[feature_columns]).astype(float)
    
    df_previsao.to_csv(f'EVALUATED\\previsao_final_{test_type}_{v_modelo}.csv',sep = ';', index = False)
    
    
    #-----------linear_regression-----------
    
    test_type = 'linear_regression'
    
    model = LinearRegression()
    
    model.fit(df_train[feature_columns], df_train[target_train])
    
    filename = f"MODELS\\model_{v_modelo}_{test_type}.sav"
    
    pickle.dump(model, open(filename, 'wb'))
    
    Y_Predict = model.predict(df_test[feature_columns])

    df_test['Y_Predict'] = Y_Predict#.astype(float)
    
    mse = mean_squared_error(df_test[target_train], df_test['Y_Predict'])

    r2_score = sklearn.metrics.r2_score(df_test[target_train], df_test['Y_Predict'])
    
    print(f'Test Type: {test_type}, Mean Square Error: {mse.round(2)}, R²: {r2_score.round(2)}')
    
    #predição do modelo
    
    #Y_Predict = model.predict(df_previsao[feature_columns])

    df_previsao['Y_Predict'] = model.predict(df_previsao[feature_columns]).astype(float)
    
    df_previsao.to_csv(f'EVALUATED\\previsao_final_{test_type}_{v_modelo}.csv',sep = ';', index = False)
    
    #-----------svm-----------
    
    # test_type = 'svm_svr'
    
    # model = SVR()
    
    # model.fit(df_train[feature_columns], df_train[target_train])
    
    # model.save(f"model_{v_modelo}_{test_type}")
    
    # Y_Predict = model.predict(df_test[feature_columns])

    # df_test['Y_Predict'] = Y_Predict.astype(float)
    
    # mse = mean_squared_error(df_test[target_train], df_test['Y_Predict'])

    # r2_score = sklearn.metrics.r2_score(df_test[target_train], df_test['Y_Predict'])
    
    # print(f'Test Type: {test_type}, Mean Square Error: {mse.round(2)}, R²: {r2_score.round(2)}')
    
    # #predição do modelo
    
    # Y_Predict = model.predict(df_previsao[feature_columns])

    # df_previsao['Y_Predict'] = Y_Predict.astype(float)
    
    # df_previsao.to_csv(f'EVALUATED\\previsao_final_{v_modelo}_{test_type}.csv',sep = ';', index = False)