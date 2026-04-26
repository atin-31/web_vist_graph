import streamlit as st
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.neighbors import kneighbors_graph
import numpy as np
import cv2
import joblib
from PIL import Image
import matplotlib.pyplot as plt
import gdown  # Đã thêm thư viện tải file từ Google Drive

# Import cấu trúc model từ thư viện nội bộ
from vist_graph.models import ViST_GCN 

# ==========================================
# CẤU HÌNH GIAO DIỆN & STYLE CSS
# ==========================================
st.set_page_config(page_title="ViST-Graph AI Portal", page_icon="🧬", layout="wide")

st.markdown("""
    <style>
    .header-container { 
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%); 
        padding: 2rem; 
        border-radius: 15px; 
        color: white; 
        margin-bottom: 2rem; 
    }
    .header-title { font-size: 2.5rem; font-weight: 800; margin-bottom: 0.5rem; }
    .header-subtitle { font-size: 1.1rem; opacity: 0.9; }
    
    /* Box hướng dẫn */
    .guide-box {
        background-color: var(--secondary-background-color);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="header-container">
        <div class="header-title">🧬 ViST-Graph AI Portal</div>
        <div class="header-subtitle">Hệ thống phân tích Hệ phiên mã không gian ảo từ ảnh Mô bệnh học (H&E)</div>
    </div>
""", unsafe_allow_html=True)

# ==========================================
# LOGIC HỆ THỐNG (ONLINE CLOUD MODE)
# ==========================================
@st.cache_resource(show_spinner=False)
def load_all_assets():
    # 1. ĐẢM BẢO CÁC THƯ MỤC TỒN TẠI TRÊN MÁY CHỦ
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # 2. KHAI BÁO ID GOOGLE DRIVE CỦA BẠN TẠI ĐÂY
    # Hướng dẫn: Lấy ID từ link Google Drive (VD: link có dạng .../d/1a2b3c4d5e/view thì ID là 1a2b3c4d5e)
    MODEL_DRIVE_ID = '1bSRncH0wWJki2b8ghBWIWpO0TilG_JGY'
    PCA_DRIVE_ID = '1wMMF7PxxVG5RkvfYhgGrbavKcC9mtNp8/'
    MAPPING_DRIVE_ID = '1h0UgTQqA71UCRlvsHrAyVqNeLS1UEdAu'

    model_path = 'models/best_vist_model.pth'
    pca_path = 'models/pca_model.pkl'
    mapping_path = 'data/gene_names_mapping.pkl'

    # Hàm tải file thông minh (Chỉ tải nếu file chưa tồn tại)
    def download_from_gdrive(file_id, output_path):
        if not os.path.exists(output_path) and file_id != 'ĐIỀN_ID_FILE_...':
            url = f'https://drive.google.com/uc?id={file_id}'
            print(f"Đang tải {output_path} từ Google Drive...")
            gdown.download(url, output_path, quiet=False)

    # Thực thi tải file
    download_from_gdrive(MODEL_DRIVE_ID, model_path)
    download_from_gdrive(PCA_DRIVE_ID, pca_path)
    download_from_gdrive(MAPPING_DRIVE_ID, mapping_path)

    # 3. KHỞI TẠO MÔ HÌNH NHƯ BÌNH THƯỜNG
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval()

    gcn = ViST_GCN(in_feats=2048, hidden_feats=256, out_feats=50)
    if os.path.exists(model_path):
        gcn.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
    gcn.eval()

    pca_obj = joblib.load(pca_path) if os.path.exists(pca_path) else None
    gene_map = joblib.load(mapping_path) if os.path.exists(mapping_path) else None
    
    return resnet, gcn, pca_obj, gene_map

# Gọi hàm load với hiệu ứng tải cho người dùng biết
with st.spinner('🔄 Máy chủ đang chuẩn bị hệ thống AI (có thể mất ít phút ở lần chạy đầu tiên để tải dữ liệu)...'):
    resnet, gcn_model, pca_model, gene_mapping = load_all_assets()

def run_pipeline(image_pil, grid_size=6):
    patch_size = 224
    img_square = image_pil.resize((patch_size * grid_size, patch_size * grid_size))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    features, coords = [], []
    for i in range(grid_size):
        for j in range(grid_size):
            patch = img_square.crop((j*patch_size, i*patch_size, (j+1)*patch_size, (i+1)*patch_size))
            with torch.no_grad():
                feat = resnet(transform(patch).unsqueeze(0)).squeeze()
            features.append(feat)
            coords.append([i, j])
    x = torch.stack(features)
    A = kneighbors_graph(np.array(coords), n_neighbors=6, mode='connectivity', include_self=True)
    edge_index = torch.tensor(np.column_stack(np.where(A.toarray() == 1)).T, dtype=torch.long)
    return x, edge_index

# ==========================================
# MENU ĐIỀU HƯỚNG
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/dna.png", width=80)
    st.markdown("### 📌 Menu Điều hướng")
    menu_selection = st.radio("Chọn chức năng:", ["🔬 Phân tích dữ liệu", "📖 Hướng dẫn & Đọc hiểu"])
    
    st.markdown("---")
    st.markdown("### 📥 Dữ liệu đầu vào")
    uploaded_file = st.file_uploader("Tải lên ảnh bệnh phẩm (H&E)", type=["jpg", "png"])

# ==========================================
# TRANG 1: PHÂN TÍCH DỮ LIỆU
# ==========================================
if menu_selection == "🔬 Phân tích dữ liệu":
    if uploaded_file:
        if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
            image = Image.open(uploaded_file).convert('RGB')
            with st.status("🧠 AI đang thực hiện giải mã không gian...", expanded=False) as status:
                x, edge_index = run_pipeline(image)
                with torch.no_grad():
                    st.session_state.output = gcn_model(x, edge_index)
                st.session_state.current_file = uploaded_file.name
                status.update(label="✅ Giải mã hoàn tất!", state="complete")

        target_mg = st.select_slider("🎯 Kéo để chọn Siêu gene (Metagene) cần xem chi tiết:", options=range(50), value=0)
        
        tab1, tab2 = st.tabs(["🗺️ BẢN ĐỒ KHÔNG GIAN & ĐỊNH LƯỢNG", "🧬 GIẢI MÃ GENE CHI TIẾT"])
        
        with tab1:
            col1, col2 = st.columns(2)
            image = Image.open(uploaded_file).convert('RGB')
            
            with col1:
                st.markdown("##### Ảnh gốc H&E")
                st.image(image, use_container_width=True)
                
            with col2:
                st.markdown(f"##### Bản đồ nhiệt Metagene {target_mg}")
                vals = st.session_state.output[:, target_mg].numpy()
                norm = cv2.normalize(vals, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                heatmap = cv2.resize(norm.reshape((6, 6)), (image.size[0], image.size[1]), interpolation=cv2.INTER_CUBIC)
                heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(np.array(image), 0.5, cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB), 0.5, 0)
                st.image(overlay, use_container_width=True)

            st.markdown("---")
            
            st.markdown("##### 📊 Định lượng biểu hiện Top 5 Metagene cốt lõi")
            mean_vals = st.session_state.output.mean(dim=0)[:5].numpy()
            labels = ['PC0 (Core)', 'MG 1', 'MG 2', 'MG 3', 'MG 4']
            
            fig, ax = plt.subplots(figsize=(10, 3.5))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('gray')
            ax.spines['left'].set_color('gray')
            ax.tick_params(axis='x', colors='gray')
            ax.tick_params(axis='y', colors='gray')
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            
            colors = ['#EF4444' if v > 0 else '#3B82F6' for v in mean_vals]
            bars = ax.bar(labels, mean_vals, color=colors, width=0.6)
            ax.axhline(0, color='gray', linewidth=1.2)
            
            for bar in bars:
                y = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, y + (0.05 if y > 0 else -0.1), 
                        f"{y:.2f}", ha='center', va='bottom' if y > 0 else 'top', 
                        fontweight='bold', fontsize=10, color='gray')
            
            st.pyplot(fig, transparent=True)
        
        with tab2:
            if pca_model:
                weights = pca_model.components_[target_mg]
                top_idx = np.argsort(weights)[-10:][::-1]
                
                st.markdown(f"##### Danh sách Top 10 Gene chủ đạo của Metagene {target_mg}")
                st.info("Bảng dưới đây hiển thị các Gene có đóng góp lớn nhất vào cụm sinh học này.")
                
                gene_data = []
                for i, idx in enumerate(top_idx):
                    name = gene_mapping[idx] if gene_mapping else f"ID: {idx}"
                    gene_data.append({"Hạng": f"#{i+1}", "Tên Gene": name, "Trọng số": f"{weights[idx]:.4f}"})
                
                st.table(gene_data)
    else:
        st.info("👈 Vui lòng tải lên ảnh mô bệnh học (H&E) ở thanh bên trái để bắt đầu quá trình phân tích.")
        st.markdown("💡 *Gợi ý: Nếu bạn chưa quen với hệ thống, hãy chuyển sang tab **Hướng dẫn & Đọc hiểu** trên thanh menu.*")

# ==========================================
# TRANG 2: HƯỚNG DẪN & ĐỌC HIỂU
# ==========================================
elif menu_selection == "📖 Hướng dẫn & Đọc hiểu":
    st.markdown("## 📖 Hướng dẫn sử dụng & Đọc hiểu Kết quả")
    st.markdown("Hệ thống **ViST-Graph** sử dụng Trí tuệ Nhân tạo (GCN) để giải mã các tín hiệu sinh học ẩn sâu trong hình ảnh mô bệnh học (H&E) thông thường. Dưới đây là cách sử dụng và phiên dịch các thông số:")

    st.markdown("""<div class="guide-box">
    <h4>🚀 1. Cách thức hoạt động cơ bản</h4>
    <ul>
        <li><b>Bước 1:</b> Tải lên một bức ảnh H&E (định dạng JPG/PNG) ở thanh công cụ bên trái.</li>
        <li><b>Bước 2:</b> Trí tuệ nhân tạo sẽ tự động cắt ảnh thành các mảnh nhỏ (Patch), xây dựng mạng lưới tế bào (Graph) và dự đoán biểu hiện của 50 nhóm siêu gene (Metagene).</li>
        <li><b>Bước 3:</b> Kéo thanh trượt để xem sự phân bố của từng nhóm Metagene trên mô bệnh học.</li>
    </ul>
    </div>""", unsafe_allow_html=True)

    st.markdown("### 🗺️ Cách đọc Bản đồ không gian (Heatmap)")
    st.markdown("""
    Bản đồ nhiệt được phủ lên trên ảnh gốc H&E nhằm mục đích định vị vùng tế bào có hoạt động gene mạnh mẽ:
    - 🔴 **Vùng Đỏ/Cam (Warm colors):** Biểu thị mức độ biểu hiện gene **rất cao**. Đây thường là khu vực có mật độ tế bào ung thư dày đặc hoặc có phản ứng miễn dịch mạnh.
    - 🔵 **Vùng Xanh dương (Cold colors):** Biểu thị mức độ biểu hiện gene **thấp hoặc không có**. Thường là các vùng mô khỏe mạnh, mô nền (stroma) hoặc khoảng trống.
    """)

    st.markdown("### 📊 Ý nghĩa Biểu đồ Định lượng (Top 5 Metagene)")
    st.markdown("""
    Biểu đồ cột đánh giá tổng quan tính chất sinh học của toàn bộ vùng mô:
    - **Cột dương (Màu đỏ):** Cụm gene đó đang hoạt động mạnh (Up-regulated) trong tấm ảnh này.
    - **Cột âm (Màu xanh):** Cụm gene đó bị ức chế (Down-regulated).
    - *Lưu ý:* `PC0 (Core)` thường là nhóm Metagene bao hàm các gene nền tảng và cốt lõi nhất của khối u.
    """)

    st.markdown("### 🧬 Cách phiên dịch Bảng Giải mã Gene")
    st.markdown("""
    Bởi vì AI dự đoán theo cụm 50 "Siêu gene" (Metagene) để chống nhiễu, Tab **Giải mã Gene** sẽ phân tích ngược lại để cho bạn biết cụm Metagene đó đại diện cho những gene thực tế nào:
    - **Tên Gene:** Các gene sinh học thực tế (VD: *TP53, BRCA1, ERBB2*).
    - **Trọng số:** Trọng số càng cao (càng gần 1), gene đó càng đóng vai trò "nhạc trưởng" định hình nên màu đỏ trên Bản đồ nhiệt của Metagene đó. Bác sĩ có thể dùng danh sách này để đưa ra phác đồ điều trị đích (Targeted Therapy).
    """)
    
    st.success("🎉 Bạn đã nắm vững cách làm chủ hệ thống ViST-Graph! Hãy chuyển về tab 'Phân tích dữ liệu' để bắt đầu.")
