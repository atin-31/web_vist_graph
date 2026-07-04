import streamlit as st
import gdown
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.neighbors import kneighbors_graph
import numpy as np
import cv2
import joblib
from PIL import Image
import matplotlib.pyplot as plt

# ==========================================
# KHU VỰC 1: ĐỊNH NGHĨA KIẾN TRÚC MÔ HÌNH (CHUẨN GCN)
# Đưa thẳng vào app.py để chống lỗi thiếu module trên Cloud
# ==========================================
class ViST_GCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super().__init__()
        self.linear1 = nn.Linear(in_feats, hidden_feats)
        self.linear2 = nn.Linear(hidden_feats, out_feats)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        val = torch.ones(edge_index.size(1), device=x.device)
        adj = torch.sparse_coo_tensor(edge_index, val, (num_nodes, num_nodes))
        deg = torch.sparse.sum(adj, dim=1).to_dense()
        deg_inv = 1.0 / deg
        deg_inv[deg_inv == float('inf')] = 0
        x = self.dropout(x)
        x = self.linear1(x)
        x = torch.sparse.mm(adj, x) * deg_inv.unsqueeze(1)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = torch.sparse.mm(adj, x) * deg_inv.unsqueeze(1)
        return x

# ==========================================
# KHU VỰC 2: CẤU HÌNH GIAO DIỆN & TẢI TÀI SẢN
# ==========================================
st.set_page_config(page_title="ViST-Graph AI Portal", page_icon="🧬", layout="wide")

# CSS tương thích Dark/Light Mode và Menu chuyên nghiệp
st.markdown("""
    <style>
    .stMetric { background-color: var(--secondary-background-color); padding: 15px; border-radius: 10px; border: 1px solid rgba(128, 128, 128, 0.2); }
    .header-container { background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%); padding: 2rem; border-radius: 15px; color: white; margin-bottom: 2rem; }
    .header-title { font-size: 2.5rem; font-weight: 800; margin-bottom: 0.5rem; }
    .header-subtitle { font-size: 1.1rem; opacity: 0.9; }
    .guide-box { background-color: var(--secondary-background-color); padding: 20px; border-radius: 10px; border-left: 5px solid #3B82F6; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="header-container">
        <div class="header-title">🧬 ViST-Graph AI Portal</div>
        <div class="header-subtitle">Hệ thống phân tích Hệ phiên mã không gian ảo từ ảnh Mô bệnh học (H&E)</div>
    </div>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_all_assets():
    # 1. ResNet50
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval()

    # --- ĐIỀN ID GOOGLE DRIVE CỦA BẠN VÀO 3 BIẾN NÀY ---
    model_id = '1bSRncH0wWJki2b8ghBWIWpO0TilG_JGY'
    pca_id   = '1wMMF7PxxVG5RkvfYhgGrbavKcC9mtNp8' 
    map_id   = '1h0UgTQqA71UCRlvsHrAyVqNeLS1UEdAu'
    # ---------------------------------------------------

    # 2. Tải GCN Model
    if not os.path.exists('best_vist_model.pth'):
        gdown.download(f'https://drive.google.com/uc?id={model_id}', 'best_vist_model.pth', quiet=False)
    gcn = ViST_GCN(in_feats=2048, hidden_feats=256, out_feats=50)
    if os.path.exists('best_vist_model.pth'):
        gcn.load_state_dict(torch.load('best_vist_model.pth', map_location='cpu'))
    gcn.eval()

    # 3. Tải PCA Key
    if not os.path.exists('gene_pca_model.pkl'):
        gdown.download(f'https://drive.google.com/uc?id={pca_id}', 'gene_pca_model.pkl', quiet=False)
    pca_obj = joblib.load('gene_pca_model.pkl') if os.path.exists('gene_pca_model.pkl') else None

    # 4. Tải Gene Name Mapping
    if not os.path.exists('gene_names_mapping.pkl'):
        gdown.download(f'https://drive.google.com/uc?id={map_id}', 'gene_names_mapping.pkl', quiet=False)
    gene_map = joblib.load('gene_names_mapping.pkl') if os.path.exists('gene_names_mapping.pkl') else None
    
    return resnet, gcn, pca_obj, gene_map

# Load dữ liệu (Có spinner báo cho người dùng biết trên web)
with st.spinner('🔄 Máy chủ đang chuẩn bị hệ thống AI (có thể mất ít phút ở lần chạy đầu tiên)...'):
    resnet, gcn_model, pca_model, gene_mapping = load_all_assets()

# ==========================================
# KHU VỰC 3: HÀM XỬ LÝ ĐỒ THỊ
# ==========================================
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
# KHU VỰC 4: MENU ĐIỀU HƯỚNG & GIAO DIỆN
# ==========================================
if "current_file_bytes" not in st.session_state:
    st.session_state.current_file_bytes = None
if "output" not in st.session_state:
    st.session_state.output = None

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/dna.png", width=80)
    st.markdown("### 📌 Menu Điều hướng")
    menu_selection = st.radio("Chọn chức năng:", ["🔬 Phân tích dữ liệu", "📖 Hướng dẫn & Đọc hiểu"])
    
    st.markdown("---")
    st.markdown("### 📥 Dữ liệu đầu vào")
    uploaded_file = st.file_uploader("Tải lên ảnh bệnh phẩm (H&E)", type=["jpg", "png"])

# ----------------- TRANG 1: PHÂN TÍCH -----------------
if menu_selection == "🔬 Phân tích dữ liệu":
    if uploaded_file is None:
        st.info("👈 Vui lòng tải lên ảnh mô bệnh học (H&E) ở thanh bên trái để bắt đầu quá trình phân tích.")
        st.markdown("💡 *Gợi ý: Nếu bạn chưa quen với hệ thống, hãy chuyển sang tab **Hướng dẫn & Đọc hiểu** trên thanh menu.*")
    else:
        image = Image.open(uploaded_file).convert('RGB')
        file_bytes = uploaded_file.getvalue()
        
        # Xử lý khi có ảnh mới
        if st.session_state.current_file_bytes != file_bytes:
            with st.status("🧠 AI đang thực hiện giải mã không gian...", expanded=False) as status:
                x, edge_index = run_pipeline(image)
                with torch.no_grad():
                    st.session_state.output = gcn_model(x, edge_index)
                st.session_state.current_file_bytes = file_bytes 
                status.update(label="✅ Giải mã hoàn tất!", state="complete")

        target_mg = st.select_slider("🎯 Kéo để chọn Siêu gene (Metagene) cần xem chi tiết:", options=range(50), value=0)
        
        tab1, tab2, tab3 = st.tabs(["🗺️ BẢN ĐỒ KHÔNG GIAN & ĐỊNH LƯỢNG", "🧬 GIẢI MÃ GENE CHI TIẾT", "📝 BỆNH ÁN TỰ ĐỘNG"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
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
                st.info("Bảng dưới đây hiển thị các Gene có đóng góp lớn nhất vào cụm sinh học này (hiển thị trọn vẹn tên gene).")
                
                gene_data = []
                for i, idx in enumerate(top_idx):
                    name = gene_mapping[idx] if gene_mapping else f"ID: {idx}"
                    gene_data.append({"Hạng": f"#{i+1}", "Tên Gene": name, "Trọng số": f"{weights[idx]:.4f}"})
                
                st.table(gene_data)
            else:
                st.warning("⚠️ Đang chờ tải file giải mã Gene từ máy chủ...")

        with tab3:
            st.markdown("##### 🤖 Trợ lý Y khoa Google Gemini & Local Heuristic Engine")
            st.info("Hệ thống tự động sinh bệnh án dựa trên danh sách các gen sinh học chủ đạo. API đã được tích hợp sẵn. Nếu mất kết nối, hệ thống sẽ tự động chuyển sang chế độ Ngoại tuyến (Local Heuristic Engine).")
            
            # Gắn sẵn API Key mới nhất của bạn
            api_key = "AQ.Ab8RN6JOshWmDepaYHArJL5UynJ8M5SWqqzc6GCthm15EOEFLw"
            
            if st.button("Tạo Bệnh Án", type="primary"):
                if pca_model:
                    weights = pca_model.components_[target_mg]
                    top_idx = np.argsort(weights)[-10:][::-1]
                    gene_list = [gene_mapping[idx] if gene_mapping else f"ID: {idx}" for idx in top_idx]
                    
                    if api_key:
                        try:
                            import google.generativeai as genai
                            genai.configure(api_key=api_key)
                            model = genai.GenerativeModel('gemini-2.5-flash')
                            prompt = f"Đóng vai một bác sĩ giải phẫu bệnh chuyên nghiệp. Hãy viết một báo cáo bệnh án ngắn gọn (khoảng 150-200 chữ) dựa trên sự xuất hiện của các dấu ấn gen sinh học chủ đạo sau đây tại vùng mô ung thư vú: {', '.join(gene_list)}. Trình bày thành các gạch đầu dòng rõ ràng về ý nghĩa lâm sàng của chúng."
                            
                            with st.spinner("🤖 Gemini đang phân tích và viết bệnh án..."):
                                response = model.generate_content(prompt)
                                st.success("✅ Đã kết nối Internet & API thành công. Bệnh án được sinh bởi Google Gemini:")
                                st.write(response.text)
                        except Exception as e:
                            # HIỆN THẲNG LỖI RA MÀN HÌNH WEB NẾU CÓ SỰ CỐ
                            st.error(f"❌ LỖI KẾT NỐI GOOGLE GEMINI: {str(e)}")
                            st.warning("⚠️ Đang kích hoạt chế độ dự phòng **Local Heuristic Engine**...")
                            fallback_mode = True
                    else:
                        st.warning("⚠️ Bạn chưa nhập API Key. Đang kích hoạt chế độ ngoại tuyến **Local Heuristic Engine**...")
                        fallback_mode = True
                        
                    if 'fallback_mode' in locals() and fallback_mode:
                        st.markdown("### 📝 Báo cáo Chẩn đoán (Local Heuristic Engine)")
                        report_md = "Dựa trên phân tích từ khóa gen mục tiêu bằng cơ chế Offline, hệ thống nhận diện các dấu ấn sinh học sau tại vi môi trường khối u:\n\n"
                        for g in gene_list:
                            if "TP53" in g: report_md += f"- **{g}**: Phát hiện gen ức chế khối u (Tumor suppressor). Nguy cơ cao xuất hiện đột biến phổ biến trong ung thư vú.\n"
                            elif "BRCA" in g: report_md += f"- **{g}**: Dấu ấn liên quan mật thiết đến ung thư vú di truyền và sửa chữa DNA.\n"
                            elif "ERBB2" in g: report_md += f"- **{g} (HER2)**: Phát hiện dấu ấn tăng sinh tế bào ung thư. Đề xuất xem xét phác đồ điều trị đích kháng HER2.\n"
                            elif "KRT" in g: report_md += f"- **{g}**: Keratin marker, đặc trưng cho tế bào biểu mô khối u tại vùng này.\n"
                            elif "MKI67" in g: report_md += f"- **{g}**: Chỉ số ki-67 liên quan đến sự tăng sinh tế bào mạnh mẽ.\n"
                            elif "ESR1" in g: report_md += f"- **{g}**: Dấu ấn thụ thể Estrogen (ER+). Đề xuất xem xét liệu pháp nội tiết.\n"
                            else: report_md += f"- **{g}**: Gen có mức độ biểu hiện cao bất thường, đóng vai trò trong cấu trúc vi môi trường khối u.\n"
                        
                        report_md += "\n*Lưu ý: Báo cáo trên được tự động sinh bằng thuật toán nội bộ dựa trên cơ sở dữ liệu marker sinh học ung thư vú.*"
                        st.info(report_md)
                else:
                    st.error("Chưa tải xong dữ liệu gen, vui lòng thử lại sau.")

# ----------------- TRANG 2: HƯỚNG DẪN -----------------
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
