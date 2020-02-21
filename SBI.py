import pandas as pd
import numpy as np
from sklearn.preprocessing as MinMaxScaler as mms
from keras.models import load_model
model = load_model('sbi_model_jo_checkkkk.h5')
def sbi_result():
  today=dt.strftime(dt.today(),'%Y-%m-%d')
  df_sbi=web.DataReader('SBIN.NS','yahoo',start='2019-1-1',end=today)
  df_sbi=df_sbi.filter(['Close']).values
  scaler=mms(feature_range=(0,1))
  df_sbi=scaler.fit_transform(df_sbi)
  df_sbi_30=df_sbi[-1:,:]
  y_30=list()
  for i in range(30):
    x_30=list()
    df_sbi_30=np.reshape(df_sbi_30,(df_sbi_30.shape[0],1))
  
    x_30=df_sbi_30[-1:,:]
    x_30=np.array(x_30)
    x_30=np.reshape(x_30,(x_30.shape[1],1,1))
    x=model.predict(x_30)
    y_30.append(x)
    df_sbi_30=np.append(df_sbi_30,x)
  y_30=np.array(y_30)
  y_30=np.reshape(y_30,(y_30.shape[0],y_30.shape[1]))
  y_30=scaler.inverse_transform(y_30)
  result=np.reshape(y_30,(30))
  return result
