import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import pandas as pd
from scipy import linalg 
import sklearn
from sklearn import *
from sklearn.impute import KNNImputer
import xgboost as xgb
from sklearn import preprocessing


def percentage_err(y_pred, dtrain):
	#Error used in XGBoost to fit the data
    y_true = dtrain.get_label()
    y_min=np.minimum(np.abs(y_pred),y_true)
    err = np.median(np.abs(y_true-y_pred)/y_min)
    return 'percentage_err', err

def normalise_df(data,names=None):
	#Given a list of columns names (names), this function normalises the columns of the dataset
    if names == None:
        names = data.columns
    scaler = preprocessing.StandardScaler()# Create the Scaler object
    scaled_data = scaler.fit_transform(data[names])# Fit your data on the scaler object
    scaled_data = pd.DataFrame(scaled_data, columns=names)
    return scaled_data

def xgb_fit(X_train,y_train,params):
	#This function fits a XGBoost machine using X_train to predict y_train.
    xgb_r = xgb.XGBRegressor(**params)
    xgb_r.fit(X_train,y_train,eval_metric=percentage_err)
    return xgb_r

def try_pred(data,params,test_size):
	#This function fits a XGBoost machine to a sample of the dataset (X_train, y_train) 
	#and returns the prediction of the machine on X_test as well as the correct number of sales y_test
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data.iloc[:,data.columns != "nb_sold"], data.nb_sold,test_size = test_size) 
    xgb_r=xgb_fit(X_train,y_train,params)
    y_pred = xgb_r.predict(X_test)
    return y_test,y_pred
    



### Extraction of the data
df=pd.read_csv('Data/Data_Scientist_-_Home_Project_-_Source_2020_12_10.csv')
df.SalesDate=pd.to_datetime(df.SalesDate) 
df.RankDate=pd.to_datetime(df.RankDate) 

### Extraction of the information from the data, we start by fitting a polynomial equation of degree 2 to the daily UnitSold 
#using the daily ranks and the department ID.
reg_dpt7=np.polyfit(np.log10(df.loc[df.DepartmentId==7].Rank),np.log10(df.loc[df.DepartmentId==7].UnitSold),deg=2)
reg_dpt2978=np.polyfit(np.log10(df.loc[df.DepartmentId==2978].Rank),np.log10(df.loc[df.DepartmentId==2978].UnitSold),deg=2)

# lists of relevant information used to predict the number of sales
dpt_id=[]
std_rank=[]
nb_sold=[]
y_pred_poly=[]
med_rank=[]
lower_rank=[]
higher_rank=[]
nb_days=[]
days_density=[]
nb_diff_days=[]

list_pid_all=list(set(df.ProductId))
per=0
i=0
print("Extraction of the information:")
for pid in list_pid_all:
    i+=1
    if int(100*i/len(list_pid_all))>per:
        per=int(100*i/len(list_pid_all))
        print(per,end="%")
    for m in [10,11]:#we only consider full months, hence october and november
        tmp=df.loc[(df.ProductId==pid)&(df.RankDate.dt.month==m)]
        if len(tmp)>0:
            nb_sold.append(tmp.UnitSold.sum())
            med_rank.append(tmp.Rank.median())
            lower_rank.append(tmp.Rank.min())
            higher_rank.append(tmp.Rank.max())
            std_rank.append(tmp.Rank.std())
            dpt_id.append(df.loc[df.ProductId==pid].DepartmentId.iloc[0])

            days=tmp.RankDate.dt.day
            nb_days.append(max(days)-min(days))
            nb_diff_days.append(len(set(days)))
            days_density.append(nb_diff_days[-1]/(nb_days[-1]+1)) #this measures the "density" of the sales, the larger the density, the more sales happen on days close to one another.
            if nb_days[-1] == 0:
                days_density[-1]=0
                
            if tmp.DepartmentId.iloc[0] == 7:
                p=reg_dpt7
            else:
                p=reg_dpt2978
            y_pred_poly.append(np.sum(10**(p[2]+ np.log10(tmp.Rank)*p[1]+ np.log10(tmp.Rank)**2*p[0]))) #the polynomial prediction over the whole month 
print()

#Next we create the dataframe made of all the relevant characteristics
data=pd.DataFrame()
data["dpt_id"]=dpt_id
data["med_rank"]=med_rank
data["lower_rank"]=lower_rank
data["higher_rank"]=higher_rank
data["std_rank"]=std_rank
data.std_rank=data.std_rank.fillna(0)
data["nb_sold"]=nb_sold
data["y_pred_poly"]=y_pred_poly
data["nb_days"]=nb_days
data["days_density"]=days_density

#Next we normalise the dataframe to avoid giving to much weight to one characteristic
scaled_data=normalise_df(data)
scaled_data['dpt_id']=data.dpt_id.replace({7:-1,2978:1})


params={
    'n_estimators':300,
    'max_depth':15,
    'learning_rate':0.05,
}

test_size=0.1

y_pred_boost3=[]
y_test_boost3=[]
print("Computation of the performance of the model")
for _ in range(100):
    print(_,end="%")
    y_test,y_pred=try_pred(scaled_data,params,test_size)
    y_test=y_test*data.nb_sold.std()+data.nb_sold.mean()#we rescale the true number of sales using the inverse normalisation previously used
    y_pred=y_pred*data.nb_sold.std()+data.nb_sold.mean()#we rescale the prediction using the inverse normalisation previously used
    y_pred_boost3+=list(y_pred)
    y_test_boost3+=list(y_test)
print()
y_pred_boost3=np.asarray(y_pred_boost3)
y_min_boost3=np.minimum(y_test_boost3,y_pred_boost3)
error_boost3=np.abs(y_pred_boost3-y_test_boost3)/np.abs(y_min_boost3)#The error we measure
print("median error = ",np.median(error_boost3))
print("mean error = ",np.mean(error_boost3))