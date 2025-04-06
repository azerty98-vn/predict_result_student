import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Dự đoán điểm học sinh Việt", layout="centered")
st.title("📘 Dự đoán điểm cuối kỳ của học sinh Việt Nam")

# --- Tạo hoặc đọc dữ liệu ban đầu ---
DATA_FILE = "du_lieu_du_doan.csv"

if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
else:
    # Tạo dữ liệu ngẫu nhiên ban đầu nếu chưa có
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
    # Tính điểm giả lập
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
    df.to_csv(DATA_FILE, index=False)

# --- Huấn luyện mô hình ---
X = df.drop(columns=["diem_cuoi_ky"])
y = df["diem_cuoi_ky"]
model = LinearRegression()
model.fit(X, y)

# --- Giao diện nhập dữ liệu ---
st.subheader("🧑‍🎓 Nhập thông tin học sinh")

col1, col2 = st.columns(2)

with col1:
    gio_hoc = st.slider("📖 Giờ học mỗi ngày", 0.5, 10.0, 3.0, 0.5)
    buoi_hoc = st.slider("📅 Số buổi học mỗi tuần", 1, 7, 5)
    hoc_them = st.slider("➕ Giờ học thêm mỗi tuần", 0.0, 5.0, 1.0, 0.5)
    diem_giua_ky = st.slider("📝 Điểm giữa kỳ", 0.0, 10.0, 6.5, 0.5)

with col2:
    dien_thoai = st.slider("📱 Sử dụng điện thoại mỗi ngày (giờ)", 0.0, 8.0, 2.0, 0.5)
    ngu = st.slider("😴 Số giờ ngủ mỗi ngày", 4.0, 10.0, 7.0, 0.5)
    cang_thang = st.slider("😣 Mức độ căng thẳng (1 ít - 5 cao)", 1, 5, 3)

# --- Dự đoán ---
input_data = pd.DataFrame([{
    'gio_hoc_moi_ngay': gio_hoc,
    'so_buoi_hoc_trong_tuan': buoi_hoc,
    'gio_hoc_them': hoc_them,
    'diem_giua_ky': diem_giua_ky,
    'dien_thoai': dien_thoai,
    'ngu': ngu,
    'cang_thang': cang_thang
}])

diem_du_doan = model.predict(input_data)[0]
st.success(f"🎯 Dự đoán điểm cuối kỳ: **{diem_du_doan:.2f} điểm**")

# --- Ghi dữ liệu vào file CSV ---
input_data["diem_cuoi_ky"] = diem_du_doan
input_data.to_csv(DATA_FILE, mode='a', header=not os.path.exists(DATA_FILE), index=False)

# --- Hiển thị bảng lịch sử ---
st.subheader("📊 Dữ liệu đã thu thập")
st.dataframe(pd.read_csv(DATA_FILE).tail(10))  # Chỉ hiển thị 10 dòng gần nhất
