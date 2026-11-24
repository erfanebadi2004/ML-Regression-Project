import numpy as np
import streamlit as st
import pandas as pd
import requests

API_URL = "http://localhost:8000/predict"

family = ['AUTOMOTIVE', 'BABY CARE', 'BEAUTY', 'BEVERAGES', 'BOOKS',
          'BREAD/BAKERY', 'CELEBRATION', 'CLEANING', 'DAIRY', 'DELI', 'EGGS',
          'FROZEN FOODS', 'GROCERY I', 'GROCERY II', 'HARDWARE',
          'HOME AND KITCHEN I', 'HOME AND KITCHEN II', 'HOME APPLIANCES',
          'HOME CARE', 'LADIESWEAR', 'LAWN AND GARDEN', 'LINGERIE',
          'LIQUOR,WINE,BEER', 'MAGAZINES', 'MEATS', 'PERSONAL CARE',
          'PET SUPPLIES', 'PLAYERS AND ELECTRONICS', 'POULTRY',
          'PREPARED FOODS', 'PRODUCE', 'SCHOOL AND OFFICE SUPPLIES',
          'SEAFOOD']

city = ['Quito', 'Cayambe', 'Latacunga', 'Riobamba', 'Ibarra',
        'Santo Domingo', 'Guaranda', 'Puyo', 'Ambato', 'Guayaquil',
        'Salinas', 'Daule', 'Babahoyo', 'Quevedo', 'Playas', 'Libertad',
        'Cuenca', 'Loja', 'Machala', 'Esmeraldas', 'Manta', 'El Carmen']

state = ['Pichincha', 'Cotopaxi', 'Chimborazo', 'Imbabura',
         'Santo Domingo de los Tsachilas', 'Bolivar', 'Pastaza',
         'Tungurahua', 'Guayas', 'Santa Elena', 'Los Rios', 'Azuay', 'Loja',
         'El Oro', 'Esmeraldas', 'Manabi']

type = ['D', 'C', 'B', 'E', 'A']

st.sidebar.title("Demand forecasting system for retail store products to support supply")

with st.form("Data Input Form"):
    st.subheader("Features Input Form")
    ft_1 = st.slider("store_nbr", 0, 54)
    ft_2 = st.text_input("family")
    ft_3 = st.slider("onpromotion", 0, 1000, 0)
    ft_4 = st.text_input("city").capitalize()
    ft_5 = st.text_input("state").capitalize()
    ft_6 = st.text_input("type").capitalize()
    ft_7 = st.slider("cluster", 0, 100, 0)
    ft_8 = st.number_input("holiday", 0, 1, 0)
    ft_9 = st.number_input("year", 2013, 2017)
    ft_10 = st.slider("month", 1, 12, 1)
    ft_11 = st.slider("day", 1, 31, 1)
    ft_12 = st.slider("day_of_week", 1, 7, 1)
    ft_13 = st.number_input("is_weekend", 0, 1, 0)
    ft_14 = st.number_input("is_month_end", 0, 1, 0)

    submit = st.form_submit_button("Performing predictive analysis")
data = {
    "store_nbr": ft_1, "family": ft_2, "onpromotion": ft_3, "city": ft_4,
    "state": ft_5, "type": ft_6, "cluster": ft_7, "holiday": ft_8,
    "year": ft_9, "month": ft_10, "day": ft_11, "day_of_week": ft_12,
    "is_weekend": ft_13, "is_month_end": ft_14
}

df = pd.DataFrame([data])

placeholder = st.empty()

if submit:
    if (ft_2 in family) and (ft_4 in city) and (ft_5 in state) and (ft_6 in type):
        payload = {"samples": [
            {"store_nbr": ft_1, "family": ft_2, "onpromotion": ft_3, "city": ft_4,
            "state": ft_5, "type": ft_6, "cluster": ft_7, "holiday": ft_8,
            "year": ft_9, "month": ft_10, "day": ft_11, "day_of_week": ft_12,
            "is_weekend": ft_13, "is_month_end": ft_14}
]}

    else:
        st.error("Please fill in the non-numeric fields.")

    with st.spinner("Sending request to the container"):
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                result = response.json()
                list_val = result.get("prediction result")
                if isinstance(list_val, list):
                    extracted_value = np.round(list_val[0])

                    st.success("Prediction completed successfully.")
                    st.metric(label="finall result", value=extracted_value)
                    st.sidebar.info(extracted_value)
                    st.sidebar.dataframe(df)

            else:
                st.error(f"server error : {response.status_code}")
        except Exception as e:
            st.error("Unexpected error")