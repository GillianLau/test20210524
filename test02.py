###测试多版本控制02
# 连接mysql
import mysql.connector
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor as XGBR
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS, GridSearchCV
cnx = mysql.connector.connect(user = 'root',password = 'root',
                              host = 'localhost',
                              database = 'mflflaw',
                              port = 3306)
cursor = cnx.cursor()
sql = 'SELECT * FROM `flawdata`;'
df = pd.read_sql(sql,cnx)
df_feature = df.loc[:,['speed', 'magnet', 'liftoff', 'idod',
       'length_predict', 'width_predict', 'depth_predict', 'flawtype_predict',
       'erf_predict', 'pipeindexM', 'thicknessM', 'regulationM', 'idodM',
       'axial_L_special',
       'axial_L_gu_loca_gap', 'axial_L_special_loca_gap',
       'axial_L_inflect_loca_gap', 'axial_L_fengfeng_loca_gap',
       'axial_L_gu_ave', 'axial_L_fenggu_left', 'axial_L_fenggu_right',
       'axial_L_area_r1', 'axial_L_area_r2', 'axial_L_area_r3',
       'axial_L_energy_r1', 'axial_L_energy_r2', 'axial_L_energy_r3',
       'axial_W1_special', 'axial_W1_special_loca_gap',
       'axial_W1_gra1y_max_gap', 'axial_W1_area_c1', 'axial_W1_area_c2',
       'axial_W1_area_c3', 'axial_W1_energy_c1', 'axial_W1_energy_c2',
       'axial_W1_energy_c3', 'axial_W2_special', 'axial_W2_special_loca_gap',
       'axial_W2_gra1y_max_gap', 'axial_W2_area_c1', 'axial_W2_area_c2',
       'axial_W2_area_c3', 'axial_W2_energy_c1', 'axial_W2_energy_c2',
       'axial_W2_energy_c3', 'axial_area', 'axial_volume'] ]
df_length_label = df.loc[:,['lengthM']]
df_width_label = df.loc[:,['widthM']]
df_depth_label = df.loc[:,['depthM']]

def try_different_method(clf,x_train,y_train,filepath):
    clf.fit(x_train,y_train)
    joblib.dump(clf, filename=filepath)

def load_model(filepath,x_test):
    model = joblib.load(filepath)
    y_test = model.predict(x_test)
    return y_test
def LowerCount(b,y1,y2):
    error = y2.reshape(y2.shape[0], 1) - y1
    a = abs(error)
    num = 0
    for i in a.values:
        if i <= b:  # 可依需要修改条件
            num += 1
    percent = num / len(a)
    mserror = MSE(y1, y2)
    return percent,mserror

try_different_method(XGBR(),df_feature,df_length_label,'length.model')
try_different_method(XGBR(),df_feature,df_width_label,'width.model')
try_different_method(XGBR(),df_feature,df_depth_label,'depth.model')
y_predl = load_model('length.model',df_feature)
y_predw = load_model('width.model',df_feature)
y_predd = load_model('depth.model',df_feature)
Lans = LowerCount(10,df_length_label,y_predl)
Wans = LowerCount(15,df_width_label,y_predw)
Dans = LowerCount(1.11,df_depth_label,y_predd)


print('ok')