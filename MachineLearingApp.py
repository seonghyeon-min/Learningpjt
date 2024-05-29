import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.datasets import load_iris
import streamlit as st
import altair as alt
import time, re, json, os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
    

def set_page_config() :
    st.set_page_config(
        page_title="Machine Learnig",
        page_icon="🔥",
        layout='wide',
        initial_sidebar_state="expanded"
    )
    
alt.themes.enable("dark")


def preprocessData() :
    data = doLoadDataforLinear()
    
    data.horsepower.replace('?', np.nan, inplace=True)
    data.dropna(subset=['horsepower'], axis=0, inplace=True)
    data.horsepower = data.horsepower.astype('float')

    return data

def checkDataPlot(resource) :
    needCols = ['mpg', 'cylinders', 'horsepower' , 'weight']
    
    plotData = resource[needCols]
    
    sidebar_counter = 3
    colSeries = st.columns(sidebar_counter)    
    
    for cont, col in zip(needCols[1:], colSeries) :
        with col :
            container = st.container(border=True, height=550)
            with container :
                # mpg 는 종속변수
                fig = px.scatter(plotData, x ='mpg', y=cont, trendline="ols")
                fig.update_layout(title=f'corr between mpg & {cont}',
                                    xaxis_title = 'mpg',
                                    yaxis_title = cont,
                                    height=500)
                st.plotly_chart(fig, theme='streamlit', use_container_width=True)
                
    for cont, col in zip(needCols[1:], colSeries) :
        with col :
            container = st.container(border=True, height=550)
            with container :
                # mpg 는 종속변수
                fig = px.density_heatmap(plotData, x ='mpg', y=cont, marginal_x='histogram', marginal_y='histogram')
                fig.update_layout(title=f'corr between mpg & {cont}',
                                    xaxis_title = 'mpg',
                                    yaxis_title = cont,
                                    height=500)
                st.plotly_chart(fig, theme='streamlit', use_container_width=True)


def doSimpleLinearRegression(resource) :
    import plotly.figure_factory as ff
    needCols = ['mpg', 'cylinders', 'horsepower' , 'weight']
    data = resource[needCols]
    
    X, y = data[['weight']], data['mpg']
    
    X_train, X_test, y_train, y_test = train_test_split(X,           #독립 변수 
                                                    y,               #종속 변수
                                                    test_size=0.3,   #검증 30%   => 트레인 사이트는 70으 된다.
                                                    random_state=10) #랜덤 추출 값 -> 무자기로 10번 섞는다.
    
    # do LinearRegression 단순선형회귀모델 
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    r_square = lr.score(X_test, y_test)

    y_hat = lr.predict(X)
    
    group_labels = ['y', 'y_hat']
    colors = ['slategray', 'magenta']
    
    fig = ff.create_distplot([y, y_hat], group_labels, bin_size=5,
                                curve_type='normal',
                                colors=colors)
    
    st.plotly_chart(fig, theme='streamlit', use_container_width=True)
    
    st.text_area('model result', f'R_square : {r_square}, lr_coef : {lr.coef_},  lr_intercept : {lr.intercept_}')
    
def doPolynomialRegression(resource) :
    import plotly.figure_factory as ff
    from sklearn.preprocessing import PolynomialFeatures
    needCols = ['mpg', 'cylinders', 'horsepower' , 'weight']
    data = resource[needCols]
    
    X, y = data[['weight']], data['mpg']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
    
    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    
    pr = LinearRegression()
    pr.fit(X_train_poly, y_train)
    X_test_poly = poly.fit_transform(X_test)
    r_square = pr.score(X_test_poly, y_test)
    y_hat_test = pr.predict(X_test_poly)
    
    
    ## scatter plot and predicted data   
    scatter_Xaxis = X_train.iloc[:,0]
    # scatterFig = px.scatter(x=scatter_Xaxis, y=y_train)
    scatterFig = go.Figure()
    
    scatterFig.add_trace(go.Scatter(
        x=scatter_Xaxis, y=y_train,
        name='Train Data',
        mode='markers',
        marker_color = 'rgba(152, 0, 0, .8)'
    ))
    
    scatterFig.add_trace(go.Scatter(
        x=X_test.iloc[:,0], y=y_hat_test,
        name='predicted Data',
        mode='markers',
        marker_color='rgba(255, 182, 193, .9)'
    ))

    ################################################
    
    st.plotly_chart(scatterFig, theme='streamlit', use_container_width=True)
    
    fig = ff.create_distplot([y, y_hat_test], ['y', 'y_poly_hat'], bin_size=5,
                                curve_type='normal',
                                colors=['slategray', 'magenta'])
    
    st.plotly_chart(fig, theme='streamlit', use_container_width=True)
    
    st.text_area('model result', f'R_square : {r_square}, lr_coef : {pr.coef_},  lr_intercept : {pr.intercept_}')
    
def doMultivariateRegression(resource) :
    import plotly.figure_factory as ff
    
    needCols = ['mpg', 'cylinders', 'horsepower' , 'weight']
    data = resource[needCols]
    
    X = data[['cylinders', 'horsepower', 'weight']]
    y = data['mpg']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    r_square = lr.score(X_test, y_test)
    y_hat = lr.predict(X_test)
    
    fig = ff.create_distplot([y, y_hat], ['y', 'y_hat'], bin_size=5,
                                curve_type='normal',
                                colors=['slategray', 'magenta'])
    
    st.plotly_chart(fig, theme='streamlit', use_container_width=True)
    
    st.text_area('model result', f'R_square : {r_square}, lr_coef : {lr.coef_},  lr_intercept : {lr.intercept_}')
    

def doClassfication() :
    iris = load_iris()
    irisData = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names']+['target'])
    irisData['target'] = irisData['target'].map({0:'setosa', 1:'versicolor', 2:'virginica'})
    
    st.dataframe(irisData)

    X = irisData.iloc[:, :-1]
    y = irisData.iloc[:, [-1]]
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score
    
    models = []
    modelsResult = []
    models.append(("LR", LogisticRegression()))
    models.append(("DT", DecisionTreeClassifier()))
    models.append(("SVM", SVC()))
    models.append(("NB", GaussianNB()))
    models.append(("KNN", KNeighborsClassifier()))
    models.append(("RF", RandomForestClassifier()))
    models.append(("GB", GradientBoostingClassifier()))
    models.append(("ANN", MLPClassifier()))

    for name, model in models :
        model.fit(X, y.values.ravel())
        y_pred = model.predict(X)
        modelsResult.append(
            f"{name}'s Accuracy : {accuracy_score(y, y_pred)}"
        )
    score = '\n'.join(modelsResult)
    st.text_area('model Result', f'{score}')
    

def doNormalizationforCls(resource) :
    from sklearn import preprocessing
    X = resource[['pclass', 'age', 'sibsp', 'parch', 'female', 'male', 'town_C', 'town_Q', 'town_S']]
    y = resource['survived']
    
    X = preprocessing.StandardScaler().fit(X).transform(X)

    return X, y

def doKNNclassification(X, y) :
    from sklearn.neighbors import KNeighborsClassifier
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10) 
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_hat = knn.predict(X_test)
    
    fig = px.histogram(x=y_test.values)
    st.plotly_chart(fig, theme='streamlit', use_container_width=True)

@st.cache_data
def doLoadDataforLinear() :
    data = pd.read_csv('./resource/auto-mpg.csv', header=None)
    needCol = ['mpg','cylinders','displacement','horsepower','weight', 'acceleration','model year','origin','name'] 

    data.columns = needCol
    
    return data
        
def main() :
    set_page_config()
    st.title('Machine Learing')
    st.header('1. Linear Regression')
    
    st.subheader('1.1 EDA')
    mpgData = preprocessData()
    checkDataPlot(mpgData)
    st.divider()
    
    st.subheader('1.2 Train')
    st.text('1.2.1 simple Linear Regression')
    doSimpleLinearRegression(mpgData)
    
    st.divider()
    st.text('1.2.2 Polynomial Regression')
    doPolynomialRegression(mpgData)
    
    st.divider()
    st.text('1.2.3 Multivariate Regression')
    doMultivariateRegression(mpgData)
    
    st.divider()
    st.header('2. Classification (Iris Data)')
    
    doClassfication()


    
if __name__ == '__main__' :
    main()
    # try :
    #     main()
    # except Exception as err :
    #     st.warning(f"🚨 Error has happend while operating dashboard")
    #     st.warning(f"{err}")