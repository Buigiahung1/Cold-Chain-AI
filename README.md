# BÁO CÁO KỸ THUẬT VÀ TÀI LIỆU HƯỚNG DẪN: GIÁM SÁT CHUỖI CUNG ỨNG LẠNH (COLD CHAIN) THỜI GIAN THỰC BẰNG RANDOM FOREST MACHINE LEARNING

---

## 1. TỔNG QUAN HỆ THỐNG (SYSTEM OVERVIEW)

### 1.1. Đặt vấn đề
Chuỗi cung ứng lạnh (Cold Chain) đòi hỏi việc duy trì dải nhiệt độ và độ ẩm nghiêm ngặt trong suốt quá trình lưu kho và vận chuyển thực phẩm, dược phẩm hoặc vắc-xin. Việc đứt gãy chuỗi lạnh (Cold Chain Breach) dẫn đến hư hỏng hàng hóa và thiệt hại kinh tế lớn.

### 1.2. Giải pháp ML Real-time
Hệ thống sử dụng cảm biến IoT để thu thập đa thông số thời gian thực (Nhiệt độ, Độ ẩm, Rung động, Mở cửa kho/xe) và áp dụng mô hình **Random Forest Classifier** để:
* Phát hiện sớm các sự cố đứt gãy chuỗi lạnh (Anomaly Detection).
* Dự báo rủi ro biến chất của sản phẩm dựa trên độ dài thời gian vi phạm dải ngưỡng.
* Phân loại trạng thái vận hành: `0: Normal` (Bình thường), `1: Warning` (Cảnh báo biến động), `2: Breach` (Sự cố nghiêm trọng).

```
+-----------------------------------------------------------------------------------+
|                                 IOT SENSOR NODE                                   |
|   [Temp Sensor]      [Humidity Sensor]      [Accelerometer]     [Door Status]     |
+-----------------------------------------------------------------------------------+
                                          |
                                          v (MQTT / HTTP Stream)
+-----------------------------------------------------------------------------------+
|                         DATA PREPROCESSING & FEATURE ENG                          |
|   - Rolling Windows (Mean, Std, Min, Max)                                         |
|   - Cumulative Temperature Excursion (AUC)                                         |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        RANDOM FOREST INFERENCE ENGINE                             |
|   - Real-time Classification (Normal / Warning / Breach)                          |
|   - Feature Importance Analysis                                                   |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                         ALERT & DASHBOARD SYSTEM                                  |
|   - Push Notifications / SMS Alerts                                               |
|   - Real-time Monitoring Dashboard (Grafana / Streamlit)                          |
+-----------------------------------------------------------------------------------+
```

---

## 2. TRÍCH XUẤT ĐẶC TRƯNG (FEATURE ENGINEERING)

Dữ liệu cảm biến thô (Raw Time-Series) cần được chuyển đổi thành dạng bảng (Tabular Data) phù hợp cho mô hình Random Forest thông qua thuật toán Cửa sổ Trượt (Sliding Window):

| Tên Đặc trưng | Mô tả | Vai trò đối với Mô hình |
| :--- | :--- | :--- |
| `temp_current` | Nhiệt độ tức thời ($^\circ\text{C}$) | Giá trị gốc tại thời điểm $t$ |
| `temp_roll_mean_5m` | Giá trị trung bình trong 5 phút | Giảm nhiễu tức thời của cảm biến |
| `temp_roll_std_5m` | Độ lệch chuẩn nhiệt độ 5 phút | Đo lường độ biến động nhiệt độ |
| `temp_rate_of_change` | Tốc độ thay đổi nhiệt độ ($\Delta T / \Delta t$) | Phát hiện sự cố hỏng máy lạnh/mở cửa xe |
| `humidity_current` | Độ ẩm tương đối (%) | Giám sát điều kiện bảo quản |
| `door_open_duration` | Thời gian mở cửa liên tục (giây) | Yếu tố gây thất thoát nhiệt chính |
| `vibration_rms` | Độ rung lắc (RMS Acceleration) | Phát hiện rung chấn mạnh gây hỏng cảm biến/thiết bị |

---

## 3. THIẾT LẬP MÔI TRƯỜNG & PHỤ THUỘC (ENVIRONMENT SETUP)

Tạo tập tin `requirements.txt`:

```text
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
joblib>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

Cài đặt các thư viện:
```bash
pip install -r requirements.txt
```

---

## 4. MÃ NGUỒN PIPELINE MÁY HỌC HOÀN CHỈNH (MAIN.PY)

Mã nguồn Python dưới đây bao gồm toàn bộ quy trình: Tạo dữ liệu giả lập chuẩn time-series, trích xuất đặc trưng cửa sổ trượt, huấn luyện Random Forest, đánh giá độ chính xác và suy luận thời gian thực (Real-time Inference Loop).

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import time

# ==========================================
# 1. TẠO DỮ LIỆU GIẢ LẬP COLD CHAIN (SIMULATION)
# ==========================================
def generate_cold_chain_data(n_samples=5000):
    np.random.seed(42)
    time_index = pd.date_range(start="2026-01-01", periods=n_samples, freq="1T")
    
    # Nhiệt độ nền dao động quanh 2-8 độ C (Dải chuẩn vắc-xin/thực phẩm)
    base_temp = 4.0 + np.sin(np.linspace(0, 50, n_samples)) * 1.5
    noise = np.random.normal(0, 0.3, n_samples)
    temperature = base_temp + noise
    
    humidity = 65.0 + np.random.normal(0, 2.0, n_samples)
    door_status = np.zeros(n_samples)
    vibration = np.random.exponential(scale=0.1, size=n_samples)
    
    label = np.zeros(n_samples) # 0: Normal
    
    # Giả lập các sự cố đứt gãy chuỗi lạnh (Breach & Warning Events)
    for i in range(100, n_samples - 200, 400):
        event_type = np.random.choice(["door_open", "cooling_failure"])
        
        if event_type == "door_open":
            # Mở cửa xe trong 15 phút
            door_status[i:i+15] = 1
            temperature[i:i+15] += np.linspace(0.5, 6.0, 15)
            label[i:i+5] = 1   # Warning
            label[i+5:i+15] = 2 # Breach (Hư hỏng)
            
        elif event_type == "cooling_failure":
            # Hỏng hệ thống làm lạnh kéo dài
            temperature[i:i+40] += np.cumsum(np.random.uniform(0.2, 0.5, 40))
            label[i:i+10] = 1   # Warning
            label[i+10:i+40] = 2 # Breach
            
    df = pd.DataFrame({
        "timestamp": time_index,
        "temperature": temperature,
        "humidity": humidity,
        "door_status": door_status,
        "vibration": vibration,
        "target": label
    })
    return df

# ==========================================
# 2. TRÍCH XUẤT ĐẶC TRƯNG CỬA SỔ TRƯỢT (SLIDING WINDOW)
# ==========================================
def extract_features(df, window_size=5):
    data = df.copy()
    
    # Biến động nhiệt độ
    data['temp_roll_mean'] = data['temperature'].rolling(window=window_size).mean()
    data['temp_roll_std'] = data['temperature'].rolling(window=window_size).std()
    data['temp_roll_max'] = data['temperature'].rolling(window=window_size).max()
    data['temp_rate_of_change'] = data['temperature'].diff(periods=1)
    
    # Thời gian mở cửa tích lũy trong cửa sổ
    data['door_open_sum'] = data['door_status'].rolling(window=window_size).sum()
    
    # Rung lắc trung bình
    data['vib_roll_mean'] = data['vibration'].rolling(window=window_size).mean()
    
    # Bỏ các dòng NaN do rolling window tạo ra
    data = data.dropna().reset_index(drop=True)
    return data

# ==========================================
# 3. HUẤN LUYỆN MÔ HÌNH RANDOM FOREST
# ==========================================
class ColdChainMonitor:
    def __init__(self, n_estimators=100, max_depth=12):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        self.feature_cols = [
            'temperature', 'humidity', 'door_status', 'vibration',
            'temp_roll_mean', 'temp_roll_std', 'temp_roll_max',
            'temp_rate_of_change', 'door_open_sum', 'vib_roll_mean'
        ]

    def train(self, df):
        X = df[self.feature_cols]
        y = df['target']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("[INFO] Đang huấn luyện Random Forest Classifier...")
        self.model.fit(X_train, y_train)
        
        # Đánh giá độ chính xác
        y_pred = self.model.predict(X_test)
        print("\n=== BÁO CÁO ĐÁNH GIÁ MÔ HÌNH ===")
        print(classification_report(y_pred, y_test, target_names=["Normal", "Warning", "Breach"]))
        
        # Độ quan trọng của các đặc trưng (Feature Importance)
        importances = self.model.feature_importances_
        feature_imp = pd.Series(importances, index=self.feature_cols).sort_values(ascending=False)
        print("\n=== MỨC ĐỘ QUAN TRỌNG CỦA CÁC ĐẶC TRƯNG ===")
        print(feature_imp)
        
    def predict_realtime(self, current_features_dict):
        """
        Dự đoán trạng thái thời gian thực từ 1 Sample dữ liệu
        """
        input_df = pd.DataFrame([current_features_dict])[self.feature_cols]
        pred_class = self.model.predict(input_df)[0]
        pred_proba = self.model.predict_proba(input_df)[0]
        return pred_class, pred_proba

    def save_model(self, filepath="cold_chain_rf.pkl"):
        joblib.dump(self.model, filepath)
        print(f"[SUCCESS] Đã lưu mô hình vào: {filepath}")

# ==========================================
# 4. LUỒNG THỰC THI CHÍNH (MAIN EXECUTION)
# ==========================================
if __name__ == "__main__":
    # Bước 1: Sinh và xử lý dữ liệu
    print("[1] Khởi tạo dữ liệu cảm biến chuỗi lạnh...")
    raw_df = generate_cold_chain_data(n_samples=6000)
    processed_df = extract_features(raw_df, window_size=5)

    # Bước 2: Huấn luyện
    print("[2] Khởi chạy Pipeline ML...")
    monitor = ColdChainMonitor(n_estimators=150, max_depth=10)
    monitor.train(processed_df)

    # Bước 3: Mô phỏng Luồng Suy luận Thời gian thực (Real-time Stream)
    print("\n[3] Bắt đầu mô phỏng luồng giám sát thời gian thực (Real-time Streaming)...")
    sample_stream = processed_df.tail(10) # Lấy 10 mẫu cuối
    
    status_map = {0: "NORMAL (Bình thường)", 1: "WARNING (Cảnh báo)", 2: "BREACH (SỰ CỐ NGHÊM TRỌNG)"}
    
    for idx, row in sample_stream.iterrows():
        sample_dict = row.to_dict()
        status, proba = monitor.predict_realtime(sample_dict)
        
        timestamp = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        temp = row['temperature']
        
        print(f"[{timestamp}] Temp: {temp:.2f}°C | Status: {status_map[status]} | Proba: {np.max(proba)*100:.1f}%")
        time.sleep(0.5) # Giả lập độ trễ truyền dữ liệu
```

---

## 5. CƠ CHẾ CẢNH BÁO VÀ NGƯỠNG RỦI RO (ALERTING & THRESHOLDS)

Hệ thống đưa ra các hành động tự động dựa trên đầu ra từ Random Forest Classifier:

1. **Trạng thái `0: NORMAL` (Xanh):**
   * **Đầu ra:** Xác suất `Normal` $> 0.85$.
   * **Hành động:** Ghi nhận log dữ liệu vào Database (TimescaleDB / InfluxDB).

2. **Trạng thái `1: WARNING` (Vàng):**
   * **Đầu ra:** Nhiệt độ dao động tiệm cận vùng nguy hiểm ($7^\circ\text{C} - 8^\circ\text{C}$) hoặc tốc độ tăng nhiệt $\Delta T > 0.5^\circ\text{C}/\text{phút}$.
   * **Hành động:** Kích hoạt cảnh báo màn hình Dashboard, gửi thông báo đẩy (Push Notification) cho tài xế/quản lý kho kiểm tra cửa hoặc thiết bị làm lạnh.

3. **Trạng thái `2: BREACH` (Đỏ):**
   * **Đầu ra:** Nhiệt độ vượt dải $> 8^\circ\text{C}$ kéo dài trên 10 phút hoặc mô hình dự báo rủi ro biến chất hỏng hàng.
   * **Hành động:** Gửi SMS/Gọi điện tự động cho Quản lý chất lượng (QA Manager), gắn nhãn kiểm định lô hàng (Quarantine Flag) khi về đến kho nhận.

---

## 6. KHUYẾN NGHỊ TRIỂN KHAI THỰC TẾ (PRODUCTION DEPLOYMENT)

* **Kiến trúc Edge Computing:** Đóng gói mô hình bằng `ONNX Runtime` để triển khai trực tiếp lên thiết bị Edge Gateway đặt trên xe container lạnh (Nvidia Jetson / Raspberry Pi) giúp hoạt động kể cả khi mất kết nối Internet.
* **Tích hợp IoT Broker:** Dùng **EMQX / Mosquitto MQTT Broker** để thu thập dữ liệu với độ trễ cực thấp (Low-latency telemetry).
* **Tái huấn luyện tự động (Continuous Training):** Đặt lịch tái huấn luyện mô hình (Retrain Pipeline) hàng tuần trên Cloud khi tích lũy thêm các kịch bản sự cố mới.# Cold-Chain-AI
