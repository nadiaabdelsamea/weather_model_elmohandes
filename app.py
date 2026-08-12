import streamlit as st
import pandas as pd
import joblib

# ============ إعدادات الصفحة ============
st.set_page_config(page_title="Weather Temperature Predictor", page_icon="🌦️", layout="centered")

st.title("🌦️ توقع درجة الحرارة (Decision Tree Model)")
st.write("املأ البيانات التالية وسيتم توقع Temperature (C) بناءً على موديل Decision Tree المدرَّب.")

# ============ تحميل الموديل ============
@st.cache_resource
def load_model():
    return joblib.load("weather_model_DT.pkl")

model = load_model()

# ترتيب الأعمدة يجب أن يطابق ترتيب التدريب تمامًا
FEATURE_ORDER = [
    "Precip Type",
    "Apparent Temperature (C)",
    "Humidity",
    "Wind Speed (km/h)",
    "Wind Bearing (degrees)",
    "Visibility (km)",
    "Loud Cover",
    "Pressure (millibars)",
    "month",
    "Hour",
]

# ============ واجهة إدخال البيانات ============
st.subheader("📋 أدخل بيانات الطقس")

col1, col2 = st.columns(2)

with col1:
    precip_type = st.selectbox("Precip Type", options=["rain", "snow"])
    apparent_temp = st.number_input("Apparent Temperature (C)", value=20.0, step=0.1, format="%.2f")
    humidity = st.slider("Humidity", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    wind_speed = st.number_input("Wind Speed (km/h)", value=10.0, step=0.1, format="%.2f")
    wind_bearing = st.slider("Wind Bearing (degrees)", min_value=0, max_value=360, value=180)

with col2:
    visibility = st.number_input("Visibility (km)", value=10.0, step=0.1, format="%.2f")
    loud_cover = st.number_input("Loud Cover", value=0.0, step=0.1, format="%.2f")
    pressure = st.number_input("Pressure (millibars)", value=1015.0, step=0.1, format="%.2f")
    month = st.selectbox("Month", options=list(range(1, 13)), index=0)
    hour = st.selectbox("Hour", options=list(range(0, 24)), index=12)

# تحويل Precip Type لنفس الترميز المستخدم في التدريب (rain=0, snow=1)
precip_encoded = 0 if precip_type == "rain" else 1

# ============ التوقع ============
if st.button("🔍 توقع درجة الحرارة", use_container_width=True):
    input_data = pd.DataFrame(
        [[
            precip_encoded,
            apparent_temp,
            humidity,
            wind_speed,
            wind_bearing,
            visibility,
            loud_cover,
            pressure,
            month,
            hour,
        ]],
        columns=FEATURE_ORDER,
    )

    prediction = model.predict(input_data)[0]

    st.success(f"🌡️ درجة الحرارة المتوقعة: **{prediction:.2f} °C**")

    with st.expander("عرض البيانات المُدخلة للموديل"):
        st.dataframe(input_data)

st.divider()
st.caption("الموديل: DecisionTreeRegressor — مدرّب على بيانات weatherHistory.csv")
