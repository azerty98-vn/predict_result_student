import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Dự đoán điểm học sinh",
    layout="centered",
    page_icon="📘"
)

# --- HEADER ---
st.markdown("<h1 style='text-align: center; color: #3a86ff;'>📘 Dự đoán điểm cuối kỳ của học sinh Việt Nam</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Nhập thông tin học tập để dự đoán điểm số. Mô hình dựa trên nhiều yếu tố học tập thực tế.</p>", unsafe_allow_html=True)

# --- FILE CONFIG ---
DATA_FILE = "du_lieu_du_doan.csv"

def load_or_create_data():
    if os.path.exists(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE)
            if df.empty or df.isnull().values.any():
                raise ValueError("File CSV trống hoặc chứa NaN.")
            return df
        except Exception as e:
            st.warning(f"⚠️ Không thể đọc file CSV: {e}")
    
    st.info("📂 Tạo dữ liệu mẫu để huấn luyện mô hình...")
    np.random.seed(42)
    df = pd.DataFrame({
        'gio_hoc_moi_ngay': np.random.uniform(1, 6, 100),
        'so_buoi_hoc_trong_tuan': np.random.randint(3, 7, 100),
        'gio_hoc_them': np.random.uniform(0, 3, 100),
        'diem_giua_ky': np.random.uniform(4, 9, 100),
        'dien_thoai': np.random.uniform(1, 6, 100),
        'ngu': np.random.uniform(5, 9, 100),
        'cang_thang': np.random.randint(1, 6, 100),
    })
    df['diem_cuoi_ky'] = (
        0.5 * df['gio_hoc_moi_ngay'] +
        0.3 * df['so_buoi_hoc_trong_tuan'] +
        0.4 * df['gio_hoc_them'] +
        0.6 * df['diem_giua_ky'] -
        0.3 * df['dien_thoai'] +
        0.2 * df['ngu'] -
        0.2 * df['cang_thang'] +
        np.random.normal(0, 0.5, 100)
    )
    return df

df = load_or_create_data()
df = df.dropna()

if df.empty:
    st.error("❌ Dữ liệu không hợp lệ. Không thể huấn luyện mô hình.")
    st.stop()

# --- TRAIN MODEL ---
X = df.drop(columns=["diem_cuoi_ky"])
y = df["diem_cuoi_ky"]
model = LinearRegression()
model.fit(X, y)

# --- FORM INPUT ---
st.markdown("## 🧑‍🏫 Nhập thông tin học sinh")

with st.form("form_du_doan"):
    col1, col2 = st.columns(2)
    with col1:
        gio_hoc = st.slider("📖 Giờ học mỗi ngày", 0.5, 10.0, 3.0, 0.5)
        buoi_hoc = st.slider("📅 Số buổi học mỗi tuần", 1, 7, 5)
        hoc_them = st.slider("➕ Giờ học thêm", 0.0, 5.0, 1.0, 0.5)
        diem_giua_ky = st.slider("📝 Điểm giữa kỳ", 0.0, 10.0, 6.5, 0.5)
    with col2:
        dien_thoai = st.slider("📱 Giờ sử dụng điện thoại", 0.0, 8.0, 2.0, 0.5)
        ngu = st.slider("😴 Giờ ngủ mỗi ngày", 4.0, 10.0, 7.0, 0.5)
        cang_thang = st.slider("😖 Mức độ căng thẳng (1–5)", 1, 5, 3)
    
    submitted = st.form_submit_button("🎯 Dự đoán điểm số")

if submitted:
    input_data = pd.DataFrame([{
        'gio_hoc_moi_ngay': gio_hoc,
        'so_buoi_hoc_trong_tuan': buoi_hoc,
        'gio_hoc_them': hoc_them,
        'diem_giua_ky': diem_giua_ky,
        'dien_thoai': dien_thoai,
        'ngu': ngu,
        'cang_thang': cang_thang
    }])
    
    prediction = model.predict(input_data)[0]
    st.success(f"📌 Dự đoán điểm cuối kỳ: **{prediction:.2f} điểm**")

    # --- Lưu vào CSV nếu local ---
    try:
        input_data["diem_cuoi_ky"] = prediction
        input_data.to_csv(DATA_FILE, mode='a', header=not os.path.exists(DATA_FILE), index=False)
    except:
        st.info("💾 Không thể ghi file vì app đang chạy online.")

# --- LỊCH SỬ DỰ ĐOÁN ---
st.markdown("## 📊 Lịch sử dự đoán gần đây")
try:
    history = pd.read_csv(DATA_FILE).tail(10)
    st.dataframe(history, use_container_width=True)
except:
    st.info("⏳ Chưa có lịch sử hoặc không đọc được dữ liệu.")
