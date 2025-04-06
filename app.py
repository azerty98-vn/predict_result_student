import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Dự đoán điểm học sinh", layout="centered")
st.title("📘 Dự đoán điểm cuối kỳ của học sinh Việt Nam")

# Đường dẫn file CSV
DATA_FILE = "du_lieu_du_doan.csv"

# ✅ Hàm load hoặc tạo dữ liệu mẫu nếu file không hợp lệ
def load_or_create_data():
    if os.path.exists(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE)
            if df.empty or df.isnull().values.any():
                raise ValueError("File CSV trống hoặc chứa NaN.")
            return df
        except Exception as e:
            st.warning(f"⚠️ Không thể đọc file CSV: {e}")
    
    # Tạo dữ liệu mẫu
    st.info("📂 Đang tạo dữ liệu mẫu để huấn luyện mô hình...")
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

# 📥 Load dữ liệu
df = load_or_create_data()

# ✅ Xử lý nếu có NaN (an toàn tuyệt đối)
df = df.dropna()
if df.empty:
    st.error("❌ Dữ liệu huấn luyện bị rỗng. Không thể tạo mô hình.")
    st.stop()

# 📊 Huấn luyện mô hình
X = df.drop(columns=["diem_cuoi_ky"])
y = df["diem_cuoi_ky"]
model = LinearRegression()
model.fit(X, y)

# 📋 Giao diện nhập dữ liệu
st.subheader("🧑‍🎓 Nhập thông tin học sinh")

col1, col2 = st.columns(2)

with col1:
    gio_hoc = st.slider("📖 Giờ học mỗi ngày", 0.5, 10.0, 3.0, 0.5)
    buoi_hoc = st.slider("📅 Số buổi học mỗi tuần", 1, 7, 5)
    hoc_them = st.slider("➕ Giờ học thêm mỗi tuần", 0.0, 5.0, 1.0, 0.5)
    diem_giua_ky = st.slider("📝 Điểm giữa kỳ", 0.0, 10.0, 6.5, 0.5)

with col2:
    dien_thoai = st.slider("📱 Giờ sử dụng điện thoại mỗi ngày", 0.0, 8.0, 2.0, 0.5)
    ngu = st.slider("😴 Số giờ ngủ mỗi ngày", 4.0, 10.0, 7.0, 0.5)
    cang_thang = st.slider("😣 Mức độ căng thẳng (1 thấp - 5 cao)", 1, 5, 3)

# 🧠 Dự đoán
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

# 💾 Ghi vào file nếu app chạy local
try:
    input_data["diem_cuoi_ky"] = diem_du_doan
    input_data.to_csv(DATA_FILE, mode='a', header=not os.path.exists(DATA_FILE), index=False)
except:
    st.info("📁 App đang chạy online – không thể lưu file lâu dài.")

# 🧾 Hiển thị bảng lịch sử gần nhất
st.subheader("📊 Lịch sử dự đoán gần đây")
try:
    history_df = pd.read_csv(DATA_FILE).tail(10)
    st.dataframe(history_df)
except:
    st.info("⏳ Chưa có lịch sử hoặc không thể đọc file.")
