import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from IPython.display import display

#"_______Gerald Clark CMSI 630 exercise2_____"


def i_func():
    # train data, dataframe_insurance
    i_data = "insurance_data/insurance.csv"
    df_i = pd.read_csv(i_data, usecols=["age", "sex", "bmi", "children", "smoker", "region", "charges"], 
                       encoding='utf-8', na_values=['?',''], index_col=False)

    df_i.to_excel("insurance_train.xlsx")
    #print(df_i)

    #shape of data
    print(df_i.shape)
    descr_i = df_i.describe()
    print(descr_i)

    #test data
    test_data_i = "insurance_data/testinputs.csv"
    df_i_test = pd.read_csv(test_data_i, usecols=["age", "sex", "bmi", "children", "smoker", "region"], 
                            encoding='utf-8', na_values=['?',''], index_col=False)
    
    df_i_test.to_excel("insurance_test.xlsx")
    
    #print(df_i_test)
    print(df_i_test.shape)

    # show heatmap of negative values, equals 0, heatmap is blacked out
    h_map = sns.heatmap(df_i.isnull(), cbar=False)
    print(h_map)

    #graph showing charges distribution
    plt.figure(figsize=(9,6))
    df_i['charges'].hist()
    plt.ylabel('Frequency')
    plt.xlabel('charges')
    plt.title('Charge Distr')
    #plt.show()

    #graph showing age distribution
    plt.figure(figsize=(9,6))
    df_i['age'].hist()
    plt.ylabel('Frequency')
    plt.xlabel('Age')
    plt.title('Age Distr')

    #scatter for bmi
    plt.scatter('bmi', 'charges', data=df_i)
    plt.title('Charges vs BMI')
    plt.xlabel('BMI')
    plt.ylabel('Charges')
    #plt.show()

    #charges vs smokers, mean


    df_i.groupby(['smoker'])['charges'].agg(['mean']).plot.bar()
    plt.ylabel('charges')
    plt.xlabel('smoker')
    plt.title('Average Charges')
    plt.show()


    # change train data to numerical data
    df_i['sex'] = df_i['sex'].apply({'male':0, 'female':1}.get)
    df_i['smoker'] = df_i['smoker'].apply({'yes':1, 'no':0}.get)
    df_i['region'] = df_i['region'].apply({'southwest':1, 'southeast':2, 'northwest':3, 'northeast':4}.get)

    df_i.to_excel("to_numeric_train.xlsx")
    print("____to numerical train data_____")
    print(df_i)

    # change test data to numeric
    df_i_test['sex'] = df_i_test['sex'].apply({'male':0, 'female':1}.get)
    df_i_test['smoker'] = df_i_test['smoker'].apply({'yes':1, 'no':0}.get)
    df_i_test['region'] = df_i_test['region'].apply({'southwest':1, 'southeast':2, 'northwest':3, 'northeast':4}.get)

    df_i_test.to_excel("to_numeric_test.xlsx")
    print("____to numerical test data_____")
    print(df_i_test)

    # catplot smoker vs charges + male/female
    sns.catplot(x="smoker", y="charges",col_wrap=3, col="sex",data= df_i, kind="box",height=5, aspect=0.8)
    #plt.show()
    #print(smoker_charges)

    # heatmap for correlations, sex has negative value
    plt.figure(figsize=(10,7))
    sns.heatmap(df_i.corr(), annot = True)
    plt.show()

    # drop sex column for it has low correlation
    X = df_i.drop(['charges', 'sex'], axis = 1)
    y = df_i.charges

    #train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
    #print("X_train shape: ", X_train.shape)
    #print("X_test shape: ", X_test.shape)
    #print("y_train shpae: ", y_train.shape)
    #print("y_test shape: ", y_test.shape)

    l_reg = LinearRegression()
    l_reg2 = l_reg.fit(X_train,y_train)
    y_pred = l_reg2.predict(X_test)

    #line plot y test to y actual
    df_a = pd.DataFrame({'Actual':y_test,'Lr':y_pred})
    plt.plot(df_a['Actual'].iloc[0:11],label='Actual')
    plt.plot(df_a['Lr'].iloc[0:11],label='Lr')
    plt.tight_layout()
    plt.legend()
    plt.show()
    
    #slope, y intercept
    c = l_reg2.intercept_
    m = l_reg2.coef_
    print("Intercept: ",c)
    print("Slope= ", m)
    #y_train_pred = l_reg.predict(X_train)
    #y_test_pred = l_reg.predict(X_test)

    #r2 score shows the accuracy of prediction(x_test) to y_test data
    print("r2 score:", r2_score(y_test, y_pred))

    plt.scatter(y_test, y_pred)
    #plt.plot(y_test, y_pred)
    plt.xlabel('Y test')
    plt.ylabel('Y pred')
    plt.show()


    sns.lmplot(x='bmi',y='charges',hue='smoker',data=df_i,aspect=1.5,height=5)
    plt.show()
    
    # drop sex column from test data
    df_i_test2 = df_i_test.drop(columns=['sex'])
    
    # cost predicition
    cost_predict = l_reg.predict(df_i_test2[:1])

    for x in cost_predict:
        #cost_predict = l_reg.predict(df_i_test2[:])
        print("Cost prediction for customer 1:", cost_predict)

    cost_predict_all = l_reg.predict(df_i_test2[:10])

    
    for i in range(0, len(cost_predict_all)):
        print("customer: ", i+1, " Predicted Charges: $", cost_predict_all[i])

    # Steps 6-10 show the accuracy of the model that is fitted using the test data against the train data. 
    # The model created then can be used to predict charges for customers of can be used to predict 
    # other outcomes mathematically using agorithms such as linear regression models...

    
    print("____complete____")
i_func()









# https://www.kaggle.com/code/kaggleashwin/linear-regression-on-insurance-dataset

# https://python.plainenglish.io/data-exploration-with-pandas-and-matplolib-69c24cb5ecee

# https://www.kaggle.com/code/kianwee/linear-regression-insurance-dataset
