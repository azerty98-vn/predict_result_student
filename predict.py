import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. Tạo dữ liệu mẫu
data = {
    'hours_per_day': [1, 2, 1.5, 3, 2.5, 4, 5, 3.5, 6, 2],
    'days_per_week': [3, 4, 2, 5, 4, 6, 7, 5, 7, 4],
    'score': [4, 6, 5, 7, 6.5, 8, 9.5, 8, 10, 6]
}

df = pd.DataFrame(data)

# 2. Chia biến đầu vào (X) và đầu ra (y)
X = df[['hours_per_day', 'days_per_week']]
y = df['score']

# 3. Chia dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Tạo và huấn luyện mô hình
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Dự đoán
y_pred = model.predict(X_test)

# 6. Đánh giá mô hình
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')

# 7. Thử dự đoán với dữ liệu mới
new_data = pd.DataFrame({
    'hours_per_day': [2.5],
    'days_per_week': [5]
})
predicted_score = model.predict(new_data)
print(f'Dự đoán điểm số: {predicted_score[0]:.2f}')
