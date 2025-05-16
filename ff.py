import cv2
import mediapipe as mp
import streamlit as st
import numpy as np
from PIL import Image

# 設置頁面配置為寬模式
st.set_page_config(layout="wide")

# 設置頁面標題
st.title("人臉網格檢測應用")
st.write("此應用程式使用MediaPipe進行人臉網格檢測")

# 選項卡設置
tab1, tab2 = st.tabs(["即時攝像頭", "圖片處理"])

# 設置MediaPipe
mpd = mp.solutions.drawing_utils
mpfm = mp.solutions.face_mesh
dspec = mpd.DrawingSpec((0, 255, 0), 1, 1)
cspec = mpd.DrawingSpec((128, 128, 128), 1, 1)
cpoint = mpfm.FACEMESH_TESSELATION

# 初始化 FaceMesh
fm = mpfm.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 處理函數
def process_image(img):
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = fm.process(imgrgb)
    
    if results.multi_face_landmarks:
        for f_landmarks in results.multi_face_landmarks:
            mpd.draw_landmarks(
                img, 
                landmark_list=f_landmarks, 
                connections=cpoint,
                landmark_drawing_spec=dspec,
                connection_drawing_spec=cspec
            )
    
    return img

# 選項卡1：即時攝像頭
with tab1:
    st.header("即時攝像頭模式")
    st.write("使用您的攝像頭拍照並進行人臉網格檢測")
    
    camera_image = st.camera_input("啟動攝像頭")
    
    if camera_image is not None:
        bytes_data = camera_image.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        # 處理圖像
        processed_img = process_image(img)
        
        # 顯示處理後的圖像
        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), caption="處理後的圖像", use_container_width=True)

# 選項卡2：圖片處理
with tab2:
    st.header("圖片處理模式")
    st.write("上傳圖片並進行人臉網格檢測")
    
    uploaded_file = st.file_uploader("選擇一張圖片", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # 將上傳的文件轉換為圖像
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # 創建兩列並排顯示
        col1, col2 = st.columns(2)
        
        # 顯示原始圖像
        with col1:
            st.subheader("原始圖像")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        # 處理圖像並顯示
        processed_img = process_image(img.copy())
        
        with col2:
            st.subheader("處理後的圖像")
            # 檢查是否有人臉標記
            results = fm.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            else:
                st.warning("未檢測到人臉")
                st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)

# 添加說明
st.markdown("""
### 使用說明
- **即時攝像頭模式**：允許瀏覽器使用您的攝像頭，對準攝像頭並拍照。
- **圖片處理模式**：上傳圖片，應用會處理並顯示帶有人臉網格的結果。

使用 MediaPipe 技術檢測人臉並生成面部網格。這對於面部識別、表情分析和 AR 應用很有用。
""") 