import streamlit as st
import pandas as pd
from urllib.parse import urlparse, parse_qs
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import requests
import os

# --- 0. HÀM TÌM LINK GỐC TỪ LINK RÚT GỌN ---
def get_real_url(url):
    if not url.startswith('http'):
        url = 'http://' + url
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        return response.url
    except:
        return url

# --- 1. HÀM TRÍCH XUẤT 10 ĐẶC TRƯNG THEO YÊU CẦU ---
def extract_features(url):
    if not url.startswith('http'):
        url = 'http://' + url
        
    parsed_url = urlparse(url)
    url_lower = url.lower()
    netloc = parsed_url.netloc
    
    # 1. Độ dài URL
    url_length = len(url)
    
    # 2. Độ dài domain
    domain_length = len(netloc)
    
    # 3. Số lượng dấu chấm (.)
    num_dots = url.count('.')
    
    # 4. Số lượng subdomain (Dựa vào số lượng dấu chấm trong phần netloc, bỏ qua www)
    # Ví dụ: login.secure.bank.com -> netloc có 3 dấu chấm -> số subdomain = số dấu chấm - 1 (nếu có www thì xử lý kỹ hơn)
    subdomains = netloc.split('.')
    if len(subdomains) > 2:
        # Loại bỏ domain chính và phần đuôi (ví dụ: com, vn)
        num_subdomains = len(subdomains) - 2
    else:
        num_subdomains = 0
        
    # 5. Số lượng ký tự đặc biệt (các ký tự không phải chữ cái, số, hoặc dấu /, ., :, -)
    # Hoặc đếm tập trung các ký tự hay dùng để obfuscate/lừa đảo: @, ?, =, &, _, %, #, !, ~, $, *
    special_chars = ['@', '?', '=', '&', '_', '%', '#', '!', '~', '$', '*']
    num_special_chars = sum(url.count(char) for char in special_chars)
    
    # 6. Số lượng chữ số
    num_digits = sum(c.isdigit() for c in url)
    
    # 7. Số lượng tham số (Parameters trong query string)
    query_params = parse_qs(parsed_url.query)
    num_parameters = len(query_params)
    
    # 8. URL có sử dụng IP hay không (1: Có, 0: Không)
    has_ip = 1 if re.search(r'\d+\.\d+\.\d+\.\d+', netloc) else 0
    
    # 9. URL có sử dụng HTTPS hay không (1: Có, 0: Không)
    has_https = 1 if parsed_url.scheme.lower() == 'https' else 0
    
    # 10. URL có chứa các từ khóa đáng ngờ hay không (1: Có, 0: Không)
    suspicious_keywords = [
        'login', 'verify', 'update', 'secure', 'account', 'banking', 
        'signin', 'confirm', 'password', 'ebayisapi', 'support', 'service', 'wallet'
    ]
    has_suspicious_keyword = 1 if any(kw in url_lower for kw in suspicious_keywords) else 0
    
    # Trả về mảng đặc trưng để đưa vào Model AI
    return [
        url_length,             # 0
        domain_length,          # 1
        num_dots,               # 2
        num_subdomains,         # 3
        num_special_chars,      # 4
        num_digits,             # 5
        num_parameters,         # 6
        has_ip,                 # 7
        has_https,              # 8
        has_suspicious_keyword  # 9
    ]

# --- 2. HÀM PHÂN TÍCH KẾT HỢP SONG SONG (AI + LUẬT CHUYÊN GIA) ---
def evaluate_url_hybrid(features, prediction_text, max_prob, url):
    """
    Hàm này kết hợp đồng thời kết quả phán đoán của AI và đánh giá chi tiết 
    từ Bộ quy tắc chuyên gia (Heuristic Rules) dựa trên 10 đặc trưng.
    """
    reasons = []
    url_lower = url.lower()
    
    # --- ĐÁNH GIÁ TỪ BỘ LUYỆT CHUYÊN GIA (HEURISTIC) ---
    if features[7] == 1: # has_ip
        reasons.append("🚨 **Dùng trực tiếp địa chỉ IP** thay vì tên miền chuẩn (Kỹ thuật ẩn danh nguy hiểm).")
    
    if features[8] == 0: # has_https
        reasons.append("⚠️ **Không sử dụng giao thức HTTPS bảo mật** (Kết nối HTTP dễ bị đánh cắp dữ liệu Man-in-the-Middle).")
        
    if features[9] == 1: # has_suspicious_keyword
        found_kw = [kw for kw in ['login', 'verify', 'update', 'secure', 'account', 'banking', 'signin', 'confirm', 'password'] if kw in url_lower]
        reasons.append(f"🚩 **Chứa từ khóa nhạy cảm:** `{', '.join(found_kw)}`. Kẻ gian thường dùng để lừa đảo thu thập thông tin (Phishing).")
        
    if features[3] > 2: # num_subdomains
        reasons.append(f"🚩 **Quá nhiều cấp Subdomain ({features[3]} cấp):** Dấu hiệu giả mạo cấu trúc tổ chức lớn (Ví dụ: `apple.com.login.verify...`).")
        
    if features[4] > 4: # num_special_chars
        reasons.append(f"🚩 **Mật độ ký tự đặc biệt cao ({features[4]} ký tự):** Dùng để che đậy đường dẫn hoặc mã hóa tham số độc hại.")
        
    if features[0] > 75: # url_length
        reasons.append(f"🚩 **Độ dài URL quá lớn ({features[0]} ký tự):** Thường nhằm nhồi nhét mã độc hoặc làm lóa mắt người dùng.")

    # --- KẾT HỢP VỚI ĐÁNH GIÁ CỦA AI ---
    ai_verdict_vi = {
        'benign': 'An toàn',
        'phishing': 'Lừa đảo (Phishing)',
        'malware': 'Chứa mã độc (Malware)',
        'defacement': 'Thay đổi giao diện (Defacement)'
    }.get(prediction_text, prediction_text.upper())

    # Nếu AI đánh giá độc hại nhưng heuristic chưa bắt được lỗi cụ thể
    if prediction_text != 'benign' and not reasons:
        reasons.append(f"⚠️ **AI phát hiện điểm bất thường tiềm ẩn:** Cấu trúc link khớp với tập dữ liệu mã độc dạng **{ai_verdict_vi}** với độ tin cậy **{max_prob:.2f}%**.")
    elif prediction_text == 'benign' and not reasons:
        reasons.append("✅ **Đánh giá đồng thuận:** Cả bộ lọc chuyên gia và AI đều xác nhận cấu trúc link tự nhiên, an toàn.")

    return reasons

# --- 3. HÀM HUẤN LUYỆN MÔ HÌNH AI ---
@st.cache_resource 
def train_model():
    possible_files = ['malicious_phish.csv', 'malicious_phish_part1.csv']
    df = None
    
    for file in possible_files:
        if os.path.exists(file):
            try:
                if 'part1' in file and os.path.exists('malicious_phish_part2.csv'):
                    df_list = [pd.read_csv('malicious_phish_part1.csv'), 
                               pd.read_csv('malicious_phish_part2.csv')]
                    if os.path.exists('malicious_phish_part3.csv'):
                        df_list.append(pd.read_csv('malicious_phish_part3.csv'))
                    df = pd.concat(df_list, ignore_index=True)
                else:
                    df = pd.read_csv(file)
                break
            except Exception as e:
                continue
                
    if df is None:
        return None, None, None
        
    # Lấy mẫu tối đa 30,000 dòng để train nhanh trên giao diện web
    if len(df) > 30000:
        df = df.sample(n=30000, random_state=42)
        
    # Trích xuất 10 đặc trưng cho tập dữ liệu training
    features = df['url'].apply(lambda x: extract_features(str(x)))
    feature_names = [
        'url_length', 'domain_length', 'num_dots', 'num_subdomains', 
        'num_special_chars', 'num_digits', 'num_parameters', 
        'has_ip', 'has_https', 'has_suspicious_keyword'
    ]
    X = pd.DataFrame(features.tolist(), columns=feature_names)
    
    le = LabelEncoder()
    y = le.fit_transform(df['type'])
    
    # Huấn luyện mô hình Random Forest
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    return model, le, feature_names

# --- 4. GIAO DIỆN WEB STREAMLIT ---
st.set_page_config(page_title="Hệ Thống Quét URL Hybrid (AI + 10 Tiêu Chí)", page_icon="🛡️", layout="centered")

st.title("🛡️ Quét URL Thông Minh: Kết Hợp AI & 10 Tiêu Chí Chuyên Gia")
st.markdown("Hệ thống kết hợp song song **Machine Learning (Random Forest)** và **Bộ lọc 10 đặc trưng cấu trúc URL** để đưa ra phán đoán toàn diện.")

with st.spinner("Đang khởi tạo và huấn luyện mô hình học máy... Vui lòng đợi trong giây lát."):
    model, label_encoder, feature_columns = train_model()

if model is None:
    st.error("❌ Không tìm thấy file dữ liệu (`malicious_phish.csv` hoặc các phần chia nhỏ). Vui lòng đặt file vào cùng thư mục với `app.py`.")
else:
    st.success("✅ Mô hình AI và Hệ thống tiêu chí đã sẵn sàng!")
    
    user_url = st.text_input("Nhập hoặc dán đường dẫn (URL) cần kiểm tra:", placeholder="https://example.com/login")
    
    if st.button("🔍 Tiến Hành Quét & Phân Tích", type="primary"):
        if not user_url.strip():
            st.warning("⚠️ Vui lòng nhập một đường link hợp lệ!")
        else:
            with st.spinner("Hệ thống đang giải mã link và phân tích đa lớp..."):
                # 1. Giải mã link rút gọn nếu có
                real_url = get_real_url(user_url)
                if real_url != user_url and real_url != ('http://' + user_url):
                    st.info(f"🔗 **Phát hiện Link Rút Gọn!** \nHệ thống đã chuyển hướng và trỏ tới địa chỉ đích thực sự: \n`{real_url}`")
                
                # 2. Trích xuất 10 đặc trưng chuẩn xác
                features_list = extract_features(real_url)
                features_df = pd.DataFrame([features_list], columns=feature_columns)
                
                # 3. Dự đoán bằng Machine Learning
                pred_encoded = model.predict(features_df)[0]
                prediction_text = label_encoder.inverse_transform([pred_encoded])[0] 
                is_malicious = (prediction_text != 'benign')
                
                probabilities = model.predict_proba(features_df)[0]
                max_prob = np.max(probabilities) * 100
                
                st.divider()
                
                # 4. Hiển thị kết quả tổng quan từ AI
                st.markdown("### 📊 Kết Quả Phán Đoán Từ AI:")
                if not is_malicious:
                    st.success(f"✅ **AN TOÀN (Benign)** - Độ tin cậy của AI: **{max_prob:.2f}%**")
                elif prediction_text == 'phishing':
                    st.error(f"🎣 **CẢNH BÁO: LỪA ĐẢO (Phishing)** - Độ tin cậy của AI: **{max_prob:.2f}%**")
                elif prediction_text == 'malware':
                    st.error(f"🦠 **CẢNH BÁO: MÃ ĐỘC (Malware)** - Độ tin cậy của AI: **{max_prob:.2f}%**")
                elif prediction_text == 'defacement':
                    st.warning(f"⚠️ **CẢNH BÁO: THAY ĐỔI GIAO DIỆN (Defacement)** - Độ tin cậy của AI: **{max_prob:.2f}%**")
                else:
                    st.error(f"🚨 **CẢNH BÁO NGUY HIỂM ({prediction_text.upper()})** - Độ tin cậy: **{max_prob:.2f}%**")
                
                # 5. Hiển thị bảng chi tiết 10 tiêu chí đã phân tích
                st.markdown("### 📋 Bảng Thống Kê 10 Tiêu Chí Trích Xuất:")
                metrics_data = {
                    "Tiêu chí đặc trưng": [
                        "1. Độ dài URL",
                        "2. Độ dài Domain",
                        "3. Số lượng dấu chấm (.)",
                        "4. Số lượng Subdomain",
                        "5. Số ký tự đặc biệt",
                        "6. Số lượng chữ số",
                        "7. Số lượng tham số",
                        "8. Sử dụng IP (1: Có, 0: Không)",
                        "9. Sử dụng HTTPS (1: Có, 0: Không)",
                        "10. Chứa từ khóa nhạy cảm (1: Có, 0: Không)"
                    ],
                    "Giá trị đo được": features_list
                }
                st.table(pd.DataFrame(metrics_data))
                
                # 6. Hiển thị phân tích lý do kết hợp song song
                st.markdown("### 🔎 Đánh Giá Chi Tiết Từ Bộ Quy Tắc Chuyên Gia:")
                reasons = evaluate_url_hybrid(features_list, prediction_text, max_prob, real_url)
                for r in reasons:
                    st.markdown(f"- {r}")