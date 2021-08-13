import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


data = pd.read_csv('https://raw.githubusercontent.com/NaveenkumarC14/Cardiovascular-Disease-EDA-and-Analytics/main/cardio_train.csv')
# separate the data into features and target
#data=data.drop('id',axis=1)
def calculate_age(days):
  days_year = 365.2425
  age = int(days // days_year)
  return age
data['age_new'] = data['age'].apply(lambda x: calculate_age(x))

features=data[['age_new','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']]
target=data['cardio']


# split the data into train and test
x_train, x_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, stratify=target
)


class StreamlitApp:

    def __init__(self):
        self.model = RandomForestClassifier()

    def train_data(self):
        self.model.fit(x_train, y_train)
        return self.model

    def construct_sidebar(self):

        cols = [col for col in features.columns]

        st.sidebar.markdown(
            '<p class="header-style">Iris Data Classification</p>',
            unsafe_allow_html=True
        )
        sepal_length = st.sidebar.selectbox(
            f"Select {cols[0]}",
            sorted(features[cols[0]].unique())
        )

        sepal_width = st.sidebar.selectbox(
            f"Select {cols[1]}",
            sorted(features[cols[1]].unique())
        )

        petal_length = st.sidebar.selectbox(
            f"Select {cols[2]}",
            sorted(features[cols[2]].unique())
        )

        petal_width = st.sidebar.selectbox(
            f"Select {cols[3]}",
            sorted(features[cols[3]].unique())
        )
        
        petal_width = st.sidebar.selectbox(
            f"Select {cols[4]}",
            sorted(features[cols[4]].unique())
        )
        petal_width = st.sidebar.selectbox(
            f"Select {cols[5]}",
            sorted(features[cols[5]].unique())
        )
        
        petal_width = st.sidebar.selectbox(
            f"Select {cols[6]}",
            sorted(features[cols[6]].unique())
        )
        petal_width = st.sidebar.selectbox(
            f"Select {cols[7]}",
            sorted(features[cols[7]].unique())
        )
        petal_width = st.sidebar.selectbox(
            f"Select {cols[8]}",
            sorted(features[cols[8]].unique())
        )
        petal_width = st.sidebar.selectbox(
            f"Select {cols[9]}",
            sorted(features[cols[9]].unique())
        )
        petal_width = st.sidebar.selectbox(
            f"Select {cols[10]}",
            sorted(features[cols[10]].unique())
        )
        
        
        
       
        values = [sepal_length, sepal_width, petal_length, petal_width,petal_length,petal_length,petal_length,petal_length,petal_length,petal_length,petal_length]

        
        if st.sidebar.button("Predict"):
           return values
          
          
           def plot_pie_chart(self, probabilities):
                fig = go.Figure(
                data=[go.Pie(labels=list(data['cardio']),values=probabilities[0])])
          fig = fig.update_traces(hoverinfo='label+percent',textinfo='value',textfont_size=15)
            return fig

           def construct_app(self):

               self.train_data()
               values = self.construct_sidebar()

               values_to_predict = np.array(values).reshape(1, -1)

               prediction = self.model.predict(values_to_predict)
               prediction_str = data.cardio[prediction[0]]
               probabilities = self.model.predict_proba(values_to_predict)

            st.markdown(
                """
                <style>
                .header-style {
                font-size:25px;
                font-family:sans-serif;
                 }
               </style>
            """,
            unsafe_allow_html=True
               )

        st.markdown(
            """
            <style>
            .font-style {
                font-size:20px;
                font-family:sans-serif;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            '<p class="header-style"> Iris Data Predictions </p>',
            unsafe_allow_html=True
        )

        column_1, column_2 = st.beta_columns(2)
        column_1.markdown(
            f'<p class="font-style" >Prediction </p>',
            unsafe_allow_html=True
        )
        column_1.write(f"{prediction_str}")

        column_2.markdown(
            '<p class="font-style" >Probability </p>',
            unsafe_allow_html=True
        )
        column_2.write(f"{probabilities[0][prediction[0]]}")

        fig = self.plot_pie_chart(probabilities)
        st.markdown(
            '<p class="font-style" >Probability Distribution</p>',
            unsafe_allow_html=True
        )
        st.plotly_chart(fig, use_container_width=True)

        return self


sa = StreamlitApp()
sa.construct_app()
