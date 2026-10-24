import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Cấu hình font tiếng Việt và hiển thị dấu trừ
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

CHECKPOINT_PATH = "vietnamese_word2vec.pt"
OUTPUT_IMAGE = "word_vectors.png"

def load_data(filepath=CHECKPOINT_PATH):
    if not os.path.exists(filepath):
        print(f"Error: '{filepath}' not found.")
        sys.exit(1)
    checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
    return checkpoint["embedding_matrix"], checkpoint["word2idx"]

def get_word_pairs():
    # Nhóm 1: Quốc gia - Thủ đô (5 cặp)
    geo_pairs = [
        ("việt_nam", "hà_nội"),
        ("pháp", "paris"),
        ("nhật_bản", "tokyo"),
        ("hàn_quốc", "seoul"),
        ("nước_anh", "london")
    ]
    # Nhóm 2: Gia đình & Vai trò (5 cặp)
    family_pairs = [
        ("bố", "mẹ"),
        ("ông", "bà"),
        ("anh_trai", "chị_gái"),
        ("con_trai", "con_gái"),
        ("vua", "hoàng_hậu")
    ]
    return geo_pairs, family_pairs

def compute_parallelism(pairs, word_to_vec):
    """Tính độ tương đồng Cosine giữa các vector hiệu để kiểm tra tính song song"""
    diff_vectors = []
    for w1, w2 in pairs:
        diff_vectors.append(word_to_vec[w2] - word_to_vec[w1])
    
    similarities = []
    n = len(pairs)
    for i in range(n):
        for j in range(i + 1, n):
            u, v = diff_vectors[i], diff_vectors[j]
            cos = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
            similarities.append(float(cos))
    return np.mean(similarities), similarities

def plot_word_embeddings():
    embeddings, word2idx = load_data()
    geo_pairs, family_pairs = get_word_pairs()
    all_pairs = geo_pairs + family_pairs

    # Lấy danh sách 20 từ duy nhất
    word_list = []
    for w1, w2 in all_pairs:
        if w1 not in word_list:
            word_list.append(w1)
        if w2 not in word_list:
            word_list.append(w2)

    # Trích xuất ma trận vector 20 x 50
    vector_matrix = np.array([embeddings[word2idx[w]] for w in word_list])

    # Giảm chiều xuống 2D bằng PCA
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(vector_matrix)
    word_to_2d = {w: coords_2d[i] for i, w in enumerate(word_list)}

    # Đánh giá định lượng tính chất song song (50D và 2D)
    word_to_50d = {w: embeddings[word2idx[w]] for w in word_list}
    mean_sim_50d, _ = compute_parallelism(geo_pairs, word_to_50d)
    mean_sim_2d, _ = compute_parallelism(geo_pairs, word_to_2d)

    print("=" * 60)
    print("KẾT QUẢ ĐO LƯỜNG TÍNH CHẤT SONG SONG (VECTOR PARALLELISM)")
    print("=" * 60)
    print(f"Tổng số từ trực quan hóa: {len(word_list)} từ (10 cặp)")
    print(f"Tỷ lệ phương sai giữ lại (PCA 2D): {pca.explained_variance_ratio_.sum() * 100:.2f}%")
    print(f"Độ tương đồng Cosine trung bình giữa các vector hiệu (Nhóm Địa lý - 50D): {mean_sim_50d:+.4f}")
    print(f"Độ tương đồng Cosine trung bình giữa các vector hiệu (Nhóm Địa lý - 2D) : {mean_sim_2d:+.4f}")
    print("=" * 60)

    # Khởi tạo khung hình trực quan hóa gồm 2 subplot trực quan
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Tùy chỉnh vị trí text để tránh đè chữ
    offsets_geo = {
        "việt_nam": (0.05, 0.02),
        "nước_anh": (0.05, 0.03),
        "nhật_bản": (-0.28, 0.01),
        "hàn_quốc": (0.05, -0.02),
        "pháp": (0.05, -0.03),
        "seoul": (-0.18, 0.02),
        "paris": (0.05, 0.02),
        "london": (0.05, -0.01),
        "tokyo": (0.05, -0.02),
        "hà_nội": (0.05, -0.03)
    }

    # --- SUBPLOT 1: Nhóm Quốc gia - Thủ đô ---
    ax1.set_title("Nhóm 1: Quan hệ Quốc gia – Thủ đô (Vector song song)", fontsize=13, fontweight='bold', pad=12)
    for country, capital in geo_pairs:
        p1 = word_to_2d[country]
        p2 = word_to_2d[capital]
        
        # Vẽ điểm tọa độ
        ax1.scatter(p1[0], p1[1], color='#1f77b4', s=90, zorder=3)
        ax1.scatter(p2[0], p2[1], color='#d62728', s=90, zorder=3)
        
        # Gắn nhãn với offset tùy chỉnh
        dx1, dy1 = offsets_geo.get(country, (0.04, 0.03))
        dx2, dy2 = offsets_geo.get(capital, (0.04, 0.03))
        ax1.text(p1[0] + dx1, p1[1] + dy1, country, fontsize=11, fontweight='semibold', color='#1f77b4')
        ax1.text(p2[0] + dx2, p2[1] + dy2, capital, fontsize=11, fontweight='semibold', color='#d62728')
        
        # Vẽ mũi tên kết nối vector
        ax1.annotate(
            "", xy=(p2[0], p2[1]), xytext=(p1[0], p1[1]),
            arrowprops=dict(arrowstyle="->", color="#555555", linestyle="--", lw=1.6, mutation_scale=15)
        )
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.set_xlabel("Principal Component 1", fontsize=11)
    ax1.set_ylabel("Principal Component 2", fontsize=11)

    offsets_fam = {
        "bố": (0.03, 0.02),
        "mẹ": (0.04, -0.05),
        "ông": (0.03, 0.02),
        "bà": (0.03, 0.02),
        "anh_trai": (0.03, 0.02),
        "chị_gái": (-0.22, 0.02),
        "con_trai": (0.03, 0.02),
        "con_gái": (0.03, 0.02),
        "vua": (0.03, -0.04),
        "hoàng_hậu": (0.03, 0.02)
    }

    # --- SUBPLOT 2: Nhóm Gia đình & Vai trò ---
    ax2.set_title("Nhóm 2: Quan hệ Gia đình – Giới tính – Vai trò", fontsize=13, fontweight='bold', pad=12)
    for m_role, f_role in family_pairs:
        p1 = word_to_2d[m_role]
        p2 = word_to_2d[f_role]
        
        # Vẽ điểm tọa độ
        ax2.scatter(p1[0], p1[1], color='#2ca02c', s=90, zorder=3)
        ax2.scatter(p2[0], p2[1], color='#9467bd', s=90, zorder=3)
        
        # Gắn nhãn
        dx1, dy1 = offsets_fam.get(m_role, (0.04, 0.03))
        dx2, dy2 = offsets_fam.get(f_role, (0.04, 0.03))
        ax2.text(p1[0] + dx1, p1[1] + dy1, m_role, fontsize=11, fontweight='semibold', color='#2ca02c')
        ax2.text(p2[0] + dx2, p2[1] + dy2, f_role, fontsize=11, fontweight='semibold', color='#9467bd')
        
        # Vẽ mũi tên kết nối vector
        ax2.annotate(
            "", xy=(p2[0], p2[1]), xytext=(p1[0], p1[1]),
            arrowprops=dict(arrowstyle="->", color="#555555", linestyle="--", lw=1.6, mutation_scale=15)
        )
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.set_xlabel("Principal Component 1", fontsize=11)
    ax2.set_ylabel("Principal Component 2", fontsize=11)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Đã lưu biểu đồ trực quan hóa thành công tại: {OUTPUT_IMAGE}")

if __name__ == "__main__":
    plot_word_embeddings()
