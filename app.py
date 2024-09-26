import joblib
import requests
from flask import Flask, jsonify, request, render_template
import os
from tensorflow.keras.models import load_model # type: ignore
import pandas as pd
from io import StringIO
import numpy as np
from datetime import date,timedelta
from tensorflow.keras import backend as K # type: ignore
from tensorflow.keras.saving import register_keras_serializable # type: ignore
@register_keras_serializable()
def r2_score(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - (ss_res / (ss_tot + K.epsilon()))
today = date.today()

yes = today - timedelta(days=1)
fifteen_days_before = today - timedelta(days=50)

print(yes)
CO2=joblib.load("co2model.pkl")
model_01 = load_model("my_model01.keras", custom_objects={'r2_score': r2_score})
model_02 = load_model("my_model02.keras", custom_objects={'r2_score': r2_score})
model_03 = load_model("model_03.keras")
comodel =joblib.load("co2model.pkl")
gdpmodel =joblib.load("gdpmodel.pkl")
allmodel =joblib.load("allmodel.pkl")
seamodel =joblib.load("seamodel.pkl")
s1=joblib.load("standard_scaler01.pkl")
s2=joblib.load("standard_scaler03.pkl")
s3=joblib.load("scalernew.pkl")
df = pd.DataFrame()
api_key="RLTYQFT7L8EETXNQSXTUAH78U"
api_link=f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/USA/{fifteen_days_before}/{yes}?unitGroup=metric&contentType=csv&include=days&key={api_key}"



response = requests.get(api_link)
if response.status_code == 200:
    csv_data = response.text
    df = pd.read_csv(StringIO(csv_data))
else:
    print(f"Error: {response.status_code}")
def GetDateArray(a):
    arr = np.zeros(12)
    if a==1:
        arr[4]=1
    elif a==2:
        arr[3]=1
    elif a==3:
        arr[7]=1
    elif a==4:
        arr[0]=1
    elif a==5:
        arr[8]=1
    elif a==6:
        arr[6]=1
    elif a==7:
        arr[5]=1
    elif a==8:
        arr[1]=1
    elif a==9:
        arr[11]=1
    elif a==10:
        arr[10]=1
    elif a==11:
        arr[9]=1
    elif a==12:
        arr[2]=1
    return arr


def getPredictions(T,TMX,TMN,HUM,SE,ST,D,U,s1,today,model_01):
    tmp = (T).reshape(1,-1)
    tmpmax = (TMX).reshape(1,-1)
    tmpmin = (TMN).reshape(1,-1)
    hum = (HUM).reshape(1,-1)
    se = (SE).reshape(1,-1)
    sr = (ST).reshape(1,-1)
    dew = (D).reshape(1,-1)
    uv = (U).reshape(1,-1)
    x=np.concatenate((tmp,GetDateArray(today.month).reshape(1,-1),tmpmax,tmpmin,hum,se,sr,dew,uv),axis=1).reshape(1,-1)
    print(GetDateArray(today.month))
    x=s1.transform(x)

    return model_01.predict(x.reshape(x.shape[0],1,x.shape[1]))
m1 = ["temp","tempmax","tempmin","humidity","solarenergy","solarradiation","dew","uvindex"]
m2=["snow","snowdepth","windgust","windspeed","winddir","precip","precipprob","precipcover"]
def todayAndTom(m1,s1,size,model_01):
    si=df[m1[0]].values.shape[0]-size 
    ans1=getPredictions(df[m1[0]].values[si:],df[m1[1]].values[si:],df[m1[2]].values[si:],df[m1[3]].values[si:],df[m1[4]].values[si:],df[m1[5]].values[si:],df[m1[6]].values[si:],df[m1[7]].values[si:],s1,today,model_01)
    TL=[]
    for i in range(8):
        el = list(df[m1[i]])
        el.append(ans1[0][i])
        TL.append(np.array(el))
    si=si+1
    ans2=getPredictions(TL[0][si:],TL[1][si:],TL[2][si:],TL[3][si:],TL[4][si:],TL[5][si:],TL[6][si:],TL[7][si:],s1,today,model_01)
    return ans1,ans2

def forecast():
    a,b=todayAndTom(m1,s1,50,model_01)
    c,d=todayAndTom(m2,s2,20,model_02)
    a=a[0]
    b=b[0]
    c=c[0]
    d=d[0]
    
    e = np.array([[a[0],c[0],c[3],c[2],a[4],a[3],a[6],c[5],c[6]]])
    f = np.array([[b[0],d[0],d[3],d[2],b[4],b[3],b[6],d[5],d[6]]])
    e=s3.transform(e)
    f=s3.transform(f)
    return getConditions(model_03.predict(e)[0]),getConditions(model_03.predict(f)[0])
def getConditions(a):
    cond=[]
    if a[0]>0.5:
        cond.append("Partially Cloudy")
    if a[1]>0.5:
        cond.append("Rain")
    if a[2]>0.5:
        cond.append("Overcast")
    if a[3]>0.5:
        cond.append("Clear")
    if a[4]>0.5:
        cond.append("Snow")
    return cond

CO2 = pd.read_csv("Co2Data.csv")
def co2(CO2,i,today):
    a=CO2["co2"].values[-15:]
    b=CO2["year"].values[-15:]
    c=list(CO2["gdp"].values[-15:])
    next_co2 = comodel.predict(a.reshape(1, -1))[0][0]
    c.append(today.year+i)
    c.append(next_co2)
    gdpans = gdpmodel.predict([c])[0][0]
    next_year = b[-1] + 1
    predicts = allmodel.predict([[gdpans,next_co2]])
    new_row = pd.DataFrame({"year": [next_year], "co2": [next_co2],"gdp":[gdpans]
                            ,"nitrous_oxide":[predicts[0][0]]
                            ,"temperature_change_from_ch4":[predicts[0][1]]
                            ,"temperature_change_from_co2":[predicts[0][2]]
                            ,"temperature_change_from_ghg":[predicts[0][3]]
                            ,"temperature_change_from_no2":[predicts[0][4]]
                            ,"total_ghg":[predicts[0][5]]
                            ,"oil_co2":[predicts[0][6]]
                                })
    CO2 = pd.concat([CO2, new_row], ignore_index=True)
    return comodel.predict(a.reshape(1,-1))[0][0],CO2,gdpans,next_year,predicts[0]

def handleCO2(CO2,today):
    for i in range(today.year-CO2["year"].values[-1]+1):
        a,CO2,b,c,d=co2(CO2,i,today)
    return CO2


sea = pd.read_csv("sea_level_NASA.csv")
def getSeapredictions(sea):
    fives = list(sea["NASA (mm)"].values[-5:])
    fives.append(sea["Year"].values[-1])
    fives.append(sea["Month"].values[-1])
    if sea["Month"].values[-1]<12:
        new_row = pd.DataFrame({"Year": [sea["Year"].values[-1]], "Month": [sea["Month"].values[-1]+1],"NASA (mm)":[seamodel.predict([fives])[0][0]]})
        sea = pd.concat([sea, new_row], ignore_index=True)
    else:
        new_row = pd.DataFrame({"Year": [sea["Year"].values[-1]+1], "Month": [1],"NASA (mm)":[seamodel.predict([fives])[0][0]]})
        sea = pd.concat([sea, new_row], ignore_index=True)
    return seamodel.predict([fives]),sea
def handleSea(sea):
    count=(today.year-sea["Year"].values[-1])*12+(today.month-sea["Month"].values[-1]+1)
    print(count)
    for i in range(count):
        a,sea=getSeapredictions(sea)
    return sea
app = Flask(__name__)
@app.route("/")
def home():
    a,b=todayAndTom(m1,s1,50,model_01)
    c,d=todayAndTom(m2,s2,20,model_02)
    f1,f2=forecast()
    sea = pd.read_csv("sea_level_NASA.csv")
    sea=handleSea(sea)
    seaData=[sea["NASA (mm)"].values[-1],sea["NASA (mm)"].values[-1]-sea["NASA (mm)"].values[-2]]
    CO2=pd.read_csv("Co2Data.csv")
    CO2=handleCO2(CO2,today)
    cdata=[CO2["gdp"].values[-1],CO2["co2"].values[-1],CO2["nitrous_oxide"].values[-1],CO2["temperature_change_from_ch4"].values[-1],CO2["temperature_change_from_co2"].values[-1],CO2["temperature_change_from_ghg"].values[-1],CO2["temperature_change_from_no2"].values[-1],CO2["total_ghg"].values[-1],CO2["oil_co2"].values[-1]]
    return render_template("home.html",td1=a[0],td2=c[0],tm1=b[0],tm2=d[0],f1=f1,f2=f2,sd=seaData,cd=cdata)
if __name__ == "__main__":
    port = int(os.environ.get("PORT",4000))
    app.run(host="0.0.0.0",port=port)
