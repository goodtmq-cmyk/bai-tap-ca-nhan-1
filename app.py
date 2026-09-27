import streamlit as st
import pandas as pd
from urllib.parse import urlparse, parse_qs
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import requests
import os

# --- DANH SÁCH TRẮNG (WHITELIST) CÁC TRANG WEB UY TÍN ---
WHITELISTED_DOMAINS = [
    'google.com', 'youtube.com', 'facebook.com', 'microsoft.com', 
    'github.com', 'apple.com', 'wikipedia.org', 'amazon.com', 
    'netflix.com', 'instagram.com', 'linkedin.com', 'zalo.me', 'vnexpress.net'
]

def is_whitelisted(url):
    if not url.startswith('http'):
        url = 'http://' + url
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    if domain.startswith('www.'):
        domain = domain[4:]
    
    # Kiểm tra xem domain có nằm trong Whitelist không
    for safe_domain in WHITELISTED_DOMAINS:
        if domain == safe_domain or domain.endswith('.' + safe_domain):
            return True
    return False

# --- 0. HÀM TÌM LINK GỐC TỪ LINK RÚT GỌN ---
def get_real_url(url):
    if not url.startswith('http'):
        url = 'http://' + url
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        return response.url
    except:
        return url

# --- 1. HÀM TRÍCH XUẤT 10 ĐẶC TRƯNG ---
def extract_features(url):
    if not url.startswith('http'):
        url = 'http://' + url
        
    parsed_url = urlparse(url)
    url_lower = url.lower()
    netloc = parsed_url.netloc
    
    url_length = len(url)
    domain_length = len(netloc)
    num_dots = url.count('.')
    
    subdomains = netloc.split('.')
    num_subdomains = len(subdomains) - 2 if len(subdomains) > 2 else 0
        
    special_chars = ['@', '?', '=', '&', '_', '%', '#', '!', '~', '$', '*']
    num_special_chars = sum(url.count(char) for char in special_chars)
    num_digits = sum(c.isdigit() for c in url)
    
    query_params = parse_qs(parsed_url.query)
    num_parameters = len(query_params)
    
    has_ip = 1 if re.search(r'\d+\.\d+\.\d+\.\d+', netloc) else 0
    has_https = 1 if parsed_url.scheme.lower() == 'https' else 0
    
    suspicious_keywords = [
        'login', 'verify', 'update', 'secure', 'account', 'banking', 
        'signin', 'confirm', 'password', 'ebayisapi', 'support', 'service', 'wallet'
    ]
    has_suspicious_keyword = 1 if any(kw in url_lower for kw in suspicious_keywords) else 0
    
    return [
        url_length, domain_length, num_dots, num_subdomains, 
        num_special_chars, num_digits, num_parameters, 
        has_ip, has_https, has_suspicious_keyword
    ]

# --- 2. HÀM PHÂN TÍCH KẾT HỢP SONG SONG ---
def evaluate_url_hybrid(features, prediction_text, max_prob, url):
    reasons = []
    url_lower = url.lower()
    
    if is_whitelisted(url):
        return ["✅ **Đây là tên miền chính thống nằm trong danh sách tin cậy (Whitelist)** của hệ thống, được bảo vệ tuyệt đối khỏi cảnh báo nhầm."]
    
    if features[7] == 1:
        reasons.append("🚨 **Dùng trực tiếp địa chỉ IP** thay vì tên miền chuẩn (Kỹ thuật ẩn danh nguy hiểm).")
    if features[8] == 0:
        reasons.append("⚠️ **Không sử dụng giao thức HTTPS bảo mật** (Kết nối HTTP dễ bị đánh cắp dữ liệu).")
    if features[9] == 1:
        found_kw = [kw for kw in ['login', 'verify', 'update', 'secure', 'account', 'banking', 'signin', 'confirm', 'password'] if kw in url_lower]
        reasons.append(f"🚩 **Chứa từ khóa nhạy cảm:** `{', '.join(found_kw)}`. Thường bị lợi dụng để lừa đảo (Phishing).")
    if features[3] > 2:
        reasons.append(f"🚩 **Quá nhiều cấp Subdomain ({features[3]} cấp):** Dấu hiệu giả mạo tổ chức lớn.")
    if features[4] > 4:
        reasons.append(f"🚩 **Mật độ ký tự đặc biệt cao ({features[4]} ký tự):** Dùng để che đậy tham số độc hại.")
    if features[0] > 75:
        reasons.append(f"🚩 **Độ dài URL quá lớn ({features[0]} ký tự):** Thường nhằm nhồi nhét mã độc.")

    ai_verdict_vi = {
        'benign': 'An toàn',
        'phishing': 'Lừa đảo (Phishing)',
        'malware': 'Chứa mã độc (Malware)',
        'defacement': 'Thay đổi giao diện (Defacement)'
    }.get(prediction_text, prediction_text.upper())

    if prediction_text != 'benign' and not reasons:
        reasons.append(f"⚠️ **AI phát hiện điểm bất thường:** Cấu trúc link khớp với tập dữ liệu dạng **{ai_verdict_vi}** (Độ tin cậy: {max_prob:.2f}%).")
    elif prediction_text == 'benign' and not reasons:
        reasons.append("✅ **Đánh giá đồng thuận:** Cả bộ lọc chuyên gia và AI đều xác nhận cấu trúc link an toàn.")

    return reasons

# --- 3. HÀM HUẤN LUYỆN MÔ HÌNH AI ---
@st.cache_resource 
def train_model(uploaded_files=None):
    df_list = []
    if uploaded_files:
        for file in uploaded_files:
            df_list.append(pd.read_csv(file))
    else:
        if os.path.exists('malicious_phish.csv'):
            df_list.append(pd.read_csv('malicious_phish.csv'))
        else:
            i = 1
            while True:
                part_file = f'malicious_phish_part{i}.csv'
                if os.path.exists(part_file):
                    df_list.append(pd.read_csv(part_file))
                    i += 1
                else:
                    break
                    
    if not df_list:
        return None, None, None
        
    df = pd.concat(df_list, ignore_index=True)
    if len(df) > 30000:
        df = df.sample(n=30000, random_state=42)
        
    features = df['url'].apply(lambda x: extract_features(str(x)))
    feature_names = [
        'url_length', 'domain_length', 'num_dots', 'num_subdomains', 
        'num_special_chars', 'num_digits', 'num_parameters', 
        'has_ip', 'has_https', 'has_suspicious_keyword'
    ]
    X = pd.DataFrame(features.tolist(), columns=feature_names)
    
    le = LabelEncoder()
    y = le.fit_transform(df['type'])
    
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    return model, le, feature_names

# --- 4. GIAO DIỆN WEB STREAMLIT ---
st.set_page_config(page_title="Hệ Thống Quét URL Hybrid (AI + 10 Tiêu Chí)", page_icon="🛡️", layout="centered")

st.title("🛡️ Quét URL Thông Minh: Kết Hợp AI & 10 Tiêu Chí Chuyên Gia")
st.markdown("Hệ thống tích hợp **Danh sách trắng (Whitelist)** chống nhận diện nhầm các trang web lớn.")

uploaded_files = st.file_uploader(
    "📂 Tải lên các file dữ liệu (malicious_phish_part1.csv, part2,...):", 
    type=['csv'], 
    accept_multiple_files=True
)

with st.spinner("Đang khởi tạo và huấn luyện mô hình học máy..."):
    model, label_encoder, feature_columns = train_model(uploaded_files)

if model is None:
    st.warning("⚠️ Vui lòng tải các file cơ sở dữ liệu lên bằng khung bên trên để tiếp tục!")
else:
    st.success("✅ Mô hình AI và Hệ thống tiêu chí đã sẵn sàng!")
    
    user_url = st.text_input("Nhập hoặc dán đường dẫn (URL) cần kiểm tra:", placeholder="https://www.google.com")
    
    if st.button("🔍 Tiến Hành Quét & Phân Tích", type="primary"):
        if not user_url.strip():
            st.warning("⚠️ Vui lòng nhập một đường link hợp lệ!")
        else:
            with st.spinner("Hệ thống đang giải mã link và phân tích đa lớp..."):
                real_url = get_real_url(user_url)
                if real_url != user_url and real_url != ('http://' + user_url):
                    st.info(f"🔗 **Phát hiện Link Rút Gọn!** \nHệ thống đã chuyển hướng tới địa chỉ thực sự: \n`{real_url}`")
                
                # KIỂM TRA WHITELIST TRƯỚC TIÊN
                if is_whitelisted(real_url):
                    prediction_text = 'benign'
                    is_malicious = False
                    max_prob = 100.0
                else:
                    features_list = extract_features(real_url)
                    features_df = pd.DataFrame([features_list], columns=feature_columns)
                    
                    pred_encoded = model.predict(features_df)[0]
                    prediction_text = label_encoder.inverse_transform([pred_encoded])[0] 
                    is_malicious = (prediction_text != 'benign')
                    
                    probabilities = model.predict_proba(features_df)[0]
                    max_prob = np.max(probabilities) * 100

                features_list = extract_features(real_url)
                
                st.divider()
                
                st.markdown("### 📊 Kết Quả Phán Đoán Từ Hệ Thống:")
                if not is_malicious:
                    st.success(f"✅ **AN TOÀN (Benign)** - Độ tin cậy: **{max_prob:.2f}%**")
                elif prediction_text == 'phishing':
                    st.error(f"🎣 **CẢNH BÁO: LỪA ĐẢO (Phishing)** - Độ tin cậy của AI: **{max_prob:.2f}%**")
                elif prediction_text == 'malware':
                    st.error(f"🦠 **CẢNH BÁO: MÃ ĐỘC (Malware)** - Độ tin cậy của AI: **{max_prob:.2f}%**")
                else:
                    st.error(f"🚨 **CẢNH BÁO NGUY HIỂM ({prediction_text.upper()})** - Độ tin cậy: **{max_prob:.2f}%**")
                
                st.markdown("### 📋 Bảng Thống Kê 10 Tiêu Chí Trích Xuất:")
                metrics_data = {
                    "Tiêu chí đặc trưng": [
                        "1. Độ dài URL", "2. Độ dài Domain", "3. Số lượng dấu chấm (.)", 
                        "4. Số lượng Subdomain", "5. Số ký tự đặc biệt", "6. Số lượng chữ số", 
                        "7. Số lượng tham số", "8. Sử dụng IP", "9. Sử dụng HTTPS", "10. Chứa từ khóa nhạy cảm"
                    ],
                    "Giá trị đo được": features_list
                }
                st.table(pd.DataFrame(metrics_data))
                
                st.markdown("### 🔎 Đánh Giá Chi Tiết Từ Bộ Quy Tắc Chuyên Gia:")
                reasons = evaluate_url_hybrid(features_list, prediction_text, max_prob, real_url)
                for r in reasons:
                    st.markdown(f"- {r}")
