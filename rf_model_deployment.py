import pandas as pd
import streamlit as st 
from sklearn.ensemble import RandomForestClassifier

st.title('Model Deployment: Insurance Fraud Detection')

st.sidebar.header('User Input Parameters')

def user_input_features():
    claim_type = st.sidebar.selectbox('claim_type',('1','2','3','4','5'))
    uninhabitable = st.sidebar.selectbox('uninhabitable',('0','1'))
    claim_amount = st.sidebar.number_input("Insert the  Claim Amount")
    coverage = st.sidebar.number_input("Insert the Coverage Amount")
    deductible = st.sidebar.number_input("Insert the deductible Amount")
    townsize = st.sidebar.selectbox('townsize',('1','2','3','4','5'))
    gender = st.sidebar.selectbox('gender',('0','1'))
    edcat = st.sidebar.selectbox('edcat',('1','2','3','4','5'))
    retire = st.sidebar.selectbox('retire',('0','1'))
    income	 = st.sidebar.number_input("Insert the Income Amount")
    marital = st.sidebar.selectbox('marital',('0','1'))
    reside = st.sidebar.selectbox('reside',('1','2','3','4','5','6','7','8','9','10'))
    primary_residence = st.sidebar.selectbox('primary_residence',('0','1'))
    data = {'claim_type':claim_type,
            'uninhabitable':uninhabitable,
            'claim_amount':claim_amount,
            'coverage':coverage,
            'deductible':deductible,
            'townsize':townsize,
            'gender':gender,
            'edcat':edcat,
            'retire':retire,
            'income':income,
            'marital':marital,
            'reside':reside,
            'primary_residence':primary_residence
            }
    features = pd.DataFrame(data,index = [0])
    return features 


    
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)

claimants = pd.read_csv("Insurance_claims.csv")
claimants.dropna()
claimants = claimants.drop(['incident_date','job_start_date','occupancy_date','policyid','policy_date','dob'], axis=1)
X = claimants.drop(['claimid','fraudulent'],axis=1)
y = claimants['fraudulent']
from imblearn.combine import SMOTETomek
smk = SMOTETomek(random_state=12)
X_res,y_res = smk.fit_resample(X,y)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_res,y_res,train_size=.8,random_state=40)
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(criterion='gini',n_estimators=80,class_weight='balanced')
rf_classifier.fit(X_train,y_train)

prediction = rf_classifier.predict(df)
prediction_proba = rf_classifier.predict_proba(df)

st.subheader('Predicted Result')
st.write('Fraud Claim' if prediction_proba[0][1] > 0.5 else 'Genuine')

st.subheader('Prediction Probability')
st.write(prediction_proba)

st.subheader('Type of claim')
st.text('1 - Wind/Hail')
st.text('2 - Water damage')
st.text('3 - Fire-Smoke')
st.text('4 - Contamination')
st.text('5 - Theft/Vandalism')

st.subheader('Property uninhabitable')
st.text('0 - No')
st.text('1 - Yes')

st.subheader('Size of hometown')
st.text('1 - > 250,000')
st.text('2 - 50,000-249,999')
st.text('3 - 10,000-49,999')
st.text('4 - 2,500-9,999')
st.text('5 - < 2,500')

st.subheader('Gender')
st.text('0 - Male')
st.text('1 - Feamle')

st.subheader('Level of education')
st.text('1 -Did not complete high school')
st.text('2 - High school degree')
st.text('3 - Some college')
st.text('4 - College degree')
st.text('5 - < Post-undergraduate degree')

st.subheader('Retired')
st.text('0 - No')
st.text('1 - Yes')


st.subheader('Marital status')
st.text('0 - Unmarried')
st.text('1 - Married')

st.subheader('Number of people in household')
st.text('1 - 10')

st.subheader('Property is primary residence')
st.text('0 -Yes')
st.text('1 -No')


