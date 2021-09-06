from flask import Flask, render_template, request, url_for, send_file, make_response, send_from_directory, redirect
import pandas as pd
import numpy as np
from scipy import stats
import logging
import datetime
import os.path
from flask import Markup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import metrics
import webbrowser
import os
from  sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from  sklearn.preprocessing import StandardScaler
import pickle
#%matplotlib inline



app = Flask(__name__)
app.config["DEBUG"] = True



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

src = os.path.join(BASE_DIR, 'processed_data_new.csv')
data = pd.read_csv(src)

#data = pd.read_csv("processed_data_new.csv")


dell=["dy"]
data=data.drop(dell,axis=1)
colmn=list(data.columns)

#feature= list(set(colmn)-set(["price"]))
#y1=data["price"].values
#x1=data[feature].values

x1=data.iloc[:,1:19].values
y1=data.iloc[:,0].values

y1=np.log(y1)
#sn= StandardScaler();
#x1=sn.fit_transform(x1)


train_x,test_x,train_y,test_y= train_test_split(x1,y1,test_size=0.3,random_state=0)

lr=LinearRegression()
model1=lr.fit(train_x,train_y)

rf=RandomForestRegressor(n_estimators=100,max_features="auto",max_depth=100,min_samples_leaf=4,min_samples_split=10,random_state=1)
model2=rf.fit(train_x,train_y)

lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
lasso_regressor=GridSearchCV(lasso,parameters)
model3=lasso_regressor.fit(train_x,train_y)

ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
ridge_regressor=GridSearchCV(ridge,parameters,cv=5)
model4=ridge_regressor.fit(train_x,train_y)

#dasdadasdad
preds1=model1.predict(test_x)
preds2=model2.predict(test_x)
preds3=model3.predict(test_x)
preds4=model4.predict(test_x)

test_preds1=model1.predict(train_x)
test_preds2=model2.predict(train_x)
test_preds3=model3.predict(train_x)
test_preds4=model4.predict(train_x)

r1test=model1.score(test_x,test_y)
r1train=model1.score(train_x,train_y)
#print(r1train,r1test)

r2test=model2.score(test_x,test_y)
r2train=model2.score(train_x,train_y)
#print(r2train,r2test)

r3test=model3.score(test_x,test_y)
r3train=model3.score(train_x,train_y)
#print(r3train,r3test)

r4test=model4.score(test_x,test_y)
r4train=model4.score(train_x,train_y)
#print(r4train,r4test)

stacked_predictions=np.column_stack((preds1,preds2,preds3,preds4))
stacked_test_predictions=np.column_stack((test_preds1, test_preds2,test_preds3, test_preds4))
meta_model=rf=RandomForestRegressor(n_estimators=100,max_features="auto",max_depth=100,min_samples_leaf=4,min_samples_split=10,random_state=1)

meta_model.fit(stacked_predictions,test_y)

#stacked_predictions
rftest=meta_model.score(stacked_predictions,test_y)
rftrain=meta_model.score(stacked_test_predictions,train_y)
#print(rftrain,rftest)



#def custom_input():





#p1,p2,p3,p4,p5,p6,p8,p9,p10,p12,p13,p14,p15,p16,p17
def get_life_expectancy(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p12,p13,p14,p15,p16,p17):
    '''    #data = pd.read_excel("C:/Users/rutuj/Desktop/BE_Project/processed_data_new.xlsx")
    #data.head()

    X = data.iloc[:,3:4]
    y = data.iloc[:,0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train) #training the algorithm
    age=age
    arr = [age]
    pd1 = pd.DataFrame(arr)
    y_pred1 = regressor.predict(pd1)
    return y_pred1'''

    #

    #

    print("p1="+p1)
    print("p2="+p2)
    print("p3="+p3)
    print("p4="+p4)
    print("p5="+p5)
    print("p6="+p6)
    print("p7="+p7)
    print("p8="+p8)
    print("p9="+p9)
    print("p10="+p10)
    print("p12="+p12)
    print("p13="+p13)
    print("p14="+p14)
    print("p15="+p15)
    print("p16="+p16)
    print("p17="+p17)
    if (p7 == 'Enter Year'):
        p7 = float(1975)
    else:
        p7 = float(p7)
    #p7 = float(2014)
    p11 = float(6)
    if (p1 == 'Enter Zipcode'):
        p1 = float(98065)
    else:
        p1 = float(p1)
    if (p2 == ''):
        p2 = float(0)
    else:
        p2 = float(p2)
    if (p3 == 'Select Bedroom'):
        p3 = float(3)
    else:
        p3 = float(p3)
    if (p4 == ''):
        p4 = float(47.5718)
    else:
        p4 = float(p4)
    if (p5 == ''):
        p5 = float(-122.2300)
    else:
        p5 = float(p5)
    if (p6 == 'Select Yes/No'):
        p6 = float(0)
    else:
        p6 = float(p6)
    if (p8 == ''):
        p8 = float(7620)
    else:
        p8 = float(p8)
    if (p9 == 'Select View'):
        p9 = float(0)
    else:
        p9 = float(p9)
    if (p10 == ''):
        p10 = float(1560)
    else:
        p10 = float(p10)
    if (p12 == 'Select Bathroom'):
        p12 = float(2.25)
    else:
        p12 = float(p12)
    if (p13 == 'Select Grade'):
        p13 = float(7)
    else:
        p13 = float(p13)
    if (p14 == 'Select Condition'):
        p14 = float(3)
    else:
        p14 = float(p14)
    if (p15 == 'Enter Year Renovation'):
        p15 = float(0)
    else:
        p15 = float(p15)
    if (p16 == 'Enter Floor'):
        p16 = float(1.5)
    else:
        p16 = float(p16)
    if (p17 == 'Enter Area'):
        p17 = float(1910)
    else:
        p17 = float(p17)
    print("After setting")
    print(p1)
    print(p2)
    print(p3)
    print(p4)
    print(p5)
    print(p6)
    print(p7)
    print(p8)
    print(p9)
    print(p10)
    print(p12)
    print(p13)
    print(p14)
    print(p15)
    print(p16)
    print(p17)

    array_our_req = [[p3,p12,p17,p8,p16,p6,p9,p14,p13,p10,p2,p7,p15,p1,p4,p5,p11]]
    print(array_our_req)
    answer1=str(p1)+'#'+str(p2)+'#'+str(p3)+'#'+str(p4)+'#'+str(p5)+'#'+str(p6)+'#'+str(p7)+'#'+str(p8)+'#'+str(p9)+'#'+str(p10)+'#'+str(p11)+'#'+str(p12)+'#'+str(p13)+'#'+str(p14)+'#'+str(p15)+'#'+str(p16)+'#'+str(p17)
    array_as_data=[]
    array_our=[[p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17]]
    print(array_our)

    #array_our=[[-2, -1,  1, -1,  0.43276959,
    #   -0.08720313, -0.69071652, -1, -0.30570773,  0.67836686,
    #   -0.50526845,  0.49977769,  5,  13, -1,
    #    0.93756346,  0.29394495]]
    #print(array_our)
    #sn= StandardScaler();
    #array_our=sn.fit_transform([array_our[0]])
    #print(array_our)

    #array_our1=[[ 3.00000e+00,  7.00000e+00,  4.73684e+01,  3.00000e+00,
    #    1.89000e+03,  0.00000e+00,  2.00000e+00,  0.00000e+00,
    #    0.00000e+00,  6.56000e+03,  2.01500e+03,  0.00000e+00,
    #   -1.22031e+02,  9.80380e+04,  3.00000e+00,  2.50000e+00,
    #    1.89000e+03]]
    #print(array_our1)

    am1=model1.predict(array_our_req)
    am2=model2.predict(array_our_req)
    am3=model3.predict(array_our_req)
    am4=model4.predict(array_our_req)

    stacked_predictions_final=np.column_stack((am1,am2,am3,am4))
    #answer = 0.0
    answer = meta_model.predict(stacked_predictions_final)
    answer = np.exp(answer)
    answer = answer[0]
    #answer = int(answer)
    answer = '{:,.2f}'.format(answer)

    print(answer)

    #answer1=str(p1)+'#'+str(p2)+'#'+str(p3)+'#'+str(p4)+'#'+str(p5)+'#'+str(p6)+'#'+str(p7)+'#'+str(p8)+'#'+str(p9)+'#'+str(p10)+'#'+str(p11)+'#'+str(p12)+'#'+str(p13)+'#'+str(p14)+'#'+str(p15)+'#'+str(p16)+'#'+str(p17)
    #answer=str(answer)+'@@@'+str(answer1)
    '''
    pickle_in = open("stacked_trial_final.pickle","rb")
    example_dict = pickle.load(pickle_in)
    print(array_our_req)
    answer_stacking = example_dict.predict([array_our_req[0]])
    print(answer_stacking)
    answer_stacking = answer_stacking[0]
    #print(answer)
    answer_stacking = np.exp(answer_stacking)
    answer_stacking = '{:,.2f}'.format(answer_stacking)
    return answer_stacking
    '''
    return answer

@app.route('/pre_processing_G26')
def view_pre():
    return send_file('templates/pre_processing_G26.pdf')
@app.route('/latex_report_G26')
def view_report():
    return send_file('templates/latex_report_G26.pdf')
@app.route('/published_work_G26')
def view_published():
    return send_file('templates/published_work_G26.pdf')

@app.route('/add_data')
def view_add_data():
    return render_template('add_data.html')

@app.route('/', methods=['POST', 'GET'])
def interact_life_expectancy():


    # select box defaults
    default_age = 'Enter Area'
    selected_age = default_age

    default_bedroom = 'Select Bedroom'
    selected_bedroom = default_bedroom

    default_bathroom = 'Select Bathroom'
    selected_bathroom = default_bathroom

    default_grade = 'Select Grade'
    selected_grade = default_grade

    default_waterfront = 'Select Yes/No'
    selected_waterfront = default_waterfront

    default_floor = 'Enter Floor'
    selected_floor = default_floor

    default_yrb = 'Enter Year'
    selected_yrb = default_yrb

    default_yrr = 'Enter Year Renovation'
    selected_yrr = default_yrr

    default_condition = 'Select Condition'
    selected_condition = default_condition

    default_view = 'Select View'
    selected_view = default_view

    default_sfa = 'Enter sqft above'
    selected_sfa = default_sfa

    default_sfb = 'Enter sqft Basement'
    selected_sfb = default_sfb

    default_sfl = 'Enter sqft Lot'
    selected_sfl = default_sfl

    default_zip = 'Enter Zipcode'
    selected_zip = default_zip

    default_lat = 'Enter Latitude'
    selected_lat = default_lat

    default_lon = 'Enter Longitude'
    selected_lon = default_lon

   # data carriers
    string_to_print = ''

    if request.method == 'POST':
        # clean up age field
        selected_age = request.form["age"]
        #if (selected_age == default_age):
        #    selected_age = float(1910)
        #else:
        #    selected_age = selected_age
        #Bedroom
        selected_bedroom = request.form["bedroom"]
        #if (selected_bedroom == default_bedroom):
        #    selected_bedroom = float(3)
        #else:
        #    selected_bedroom = selected_bedroom
        #bathroom
        selected_bathroom = request.form["bathroom"]
        #if (selected_bathroom == default_bathroom):
        #    selected_bathroom = float(2.25)
        #else:
        #    selected_bathroom = selected_bathroom
        #grade
        selected_grade = request.form["grade"]
        #if (selected_grade == default_grade):
        #    selected_grade = float(7)
        #else:
        #    selected_grade = selected_grade
        #waterfront
        selected_waterfront = request.form["waterfront"]
        #if (selected_waterfront == default_waterfront):
        #    selected_waterfront = float(0)
        #else:
        #    selected_waterfront = selected_waterfront
        #floor
        selected_floor = request.form["floor"]
        #if (selected_floor == default_floor):
        #    selected_floor = float(1.5)
        #else:
        #    selected_floor = selected_floor
        #year built
        selected_yrb = request.form["yrb"]
        #if (selected_yrb == default_yrb):
        #    selected_yrb = float(1960)
        #else:
        #    selected_yrb = selected_yrb
        #year renovated
        selected_yrr = request.form["yrr"]
        #if(selected_yrr == default_yrr):
        #    selected_yrr = float(0)
        #else:
        #    selected_yrr = selected_yrr
        #condition
        selected_condition = request.form["condition"]
        #if(selected_condition == default_condition):
        #    selected_condition = float(3)
        #else:
        #    selected_bedroom = selected_condition
        #view
        selected_view = request.form["view"]
        #if(selected_view == default_view):
        #    selected_view = float(0)
        #else:
        #    selected_view = selected_view
        #sqft above
        selected_sfa = request.form["sfa"]
        #if(selected_sfa == default_sfa):
        #    selected_sfa = float(1560)
        #else:
        #    selected_sfa = selected_sfa
        #sqft basement
        selected_sfb = request.form["sfb"]
        #if(selected_sfb == default_sfb):
        #    selected_sfb = float(0)
        #else:
        #    selected_sfb = selected_sfb
        #sqft lot
        selected_sfl = request.form["sfl"]
        #if(selected_sfl == default_sfl):
        #    selected_sfl = float(7620)
        #else:
        #    selected_sfl = selected_sfl
        # zip
        selected_zip = request.form["zip"]
        #if(selected_zip == default_zip):
        #    selected_zip = float(98065)
        #else:
        #    selected_zip = selected_zip
        #latitude
        selected_lat = request.form["lat"]
        #selected_lat = 2.2356
        #if(selected_lat == default_lat):
        #    selected_lat = float(47.5718)
        #else:
        #    selected_lat = selected_lat

        #if (selected_lat == 'Enter Latitude'):
        #    selected_lat = float(22.5)
        #selected_lat = selected_lat
        #longitude
        selected_lon = request.form["lon"]
        #if(selected_lon == default_lon):
        #    selected_lon = float(-122.2300)
        #else:
        #    selected_lon = selected_lon



        #sfb=selected_sfb,bedroom=selected_bedroom,lat=selected_lat,log=selected_lon,waterfront=selected_waterfront,sfl=selected_sfl,view=selected_view,sfa=selected_sfa,bathroom=selected_bathroom,grade=selected_grade,condition=selected_condition,yrr=selected_yrr,floor=selected_floor,age=selected_age
        #p1=selected_zip,p2=selected_sfb,p3=selected_bedroom,p4=selected_lat,p5=selected_lon,p6=selected_waterfront,p8=selected_sfl,p9=selected_view,p10=selected_sfa,p12=selected_bathroom,p13=selected_grade,p14=selected_condition,p15=selected_yrb,p16=selected_floor,p17=selected_age
        # estimate lifespan
        predicted_price = get_life_expectancy(p1=selected_zip,p2=selected_sfb,p3=selected_bedroom,p4=selected_lat,p5=selected_lon,p6=selected_waterfront,p7=selected_yrb,p8=selected_sfl,p9=selected_view,p10=selected_sfa,p12=selected_bathroom,p13=selected_grade,p14=selected_condition,p15=selected_yrr,p16=selected_floor,p17=selected_age)

        if (predicted_price is not None):
            # create output string
            string_to_print = Markup("Predicted House Price is  <font size='+10'>" + str(predicted_price) + "</font> Dollars based on previous available data!")
        else:
            string_to_print = Markup("Error! No data found for selected parameters")
            #current_time_left = 1


    return render_template('time.html',
                            string_to_print = string_to_print,
                            default_age = selected_age,
                            default_bedroom = selected_bedroom,
                            default_bathroom = selected_bathroom,
                            default_grade = selected_grade,
                            default_waterfront = selected_waterfront,
                            default_floor = selected_floor,
                            default_yrb = selected_yrb,
                            default_yrr = selected_yrr,
                            default_condition = selected_condition,
                            default_view = selected_view,
                            default_sfa = selected_sfa,
                            default_sfb = selected_sfb,
                            default_sfl = selected_sfl,
                            default_zip = selected_zip,
                            default_lat = selected_lat,
                            default_lon = selected_lon)



#if __name__ == "__main__":
#    app.run(debug=True)

