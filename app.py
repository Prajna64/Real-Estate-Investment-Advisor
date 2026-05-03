import streamlit as st
import pandas as pd
import joblib

st.title("Real Estate Investment Advisor")
st.write("Fill the details and get investment prediction plus 5-year price forecast.")
clf = joblib.load("good_investment_rf.pkl")
reg = joblib.load("future_price_rf.pkl")
feature_cols = joblib.load("feature_cols.pkl")
feature_cols_clf = [c for c in feature_cols if c != "Good_Investment"]
reg = joblib.load("future_price_rf.pkl")
feature_cols = joblib.load("feature_cols.pkl")
state = st.text_input("State", "Karnataka")
city = st.text_input("City", "Bengaluru")
bhk = st.number_input("BHK", min_value=1, max_value=10, value=3)
size = st.number_input("Size (SqFt)", min_value=200, max_value=10000, value=1200)
price = st.number_input("Current Price (Lakhs)", min_value=5.0, max_value=1000.0, value=100.0)
schools = st.number_input("Nearby Schools", min_value=0, max_value=10, value=3)
hospitals = st.number_input("Nearby Hospitals", min_value=0, max_value=10, value=2)
amenities = st.number_input("Number of Amenities", min_value=1, max_value=5, value=3)

if st.button("Predict"):
	input_dict = {col: 0 for col in feature_cols}
	if "BHK" in input_dict:
		input_dict["BHK"] = bhk
	if "Size_in_SqFt" in input_dict:
		input_dict["Size_in_SqFt"] = size
	if "Price_in_Lakhs" in input_dict:
		input_dict["Price_in_Lakhs"] = price
	if "Nearby_Schools" in input_dict:
		input_dict["Nearby_Schools"] = schools
	if "Nearby_Hospitals" in input_dict:
		input_dict["Nearby_Hospitals"] = hospitals
	if "Amenity_Count" in input_dict:
		input_dict["Amenity_Count"] = amenities
	state_col = f"State_{state}"
	city_col = f"City_{city}"
	if state_col in input_dict:
		input_dict[state_col] = 1
	if city_col in input_dict:
		input_dict[city_col] = 1
	st.write(list(input_dict.keys()))
	df_input = pd.DataFrame([input_dict], columns=feature_cols)
	df_input_clf = df_input[feature_cols_clf]
	invest_pred = clf.predict(df_input_clf)
	proba_arr = clf.predict_proba(df_input_clf) # shape(1,2)
	invest_prob = proba_arr # single float, NO float()
	future_price = reg.predict(df_input) # single float
	st.subheader("Results")
	st.write(f"Good Investment? {'Yes' if invest_pred == 1 else 'No'}")
	st.write(f"Investment probability: {invest_prob}")
	st.write(f"Estimated Price After 5 Years: {future_price} Lakhs")

