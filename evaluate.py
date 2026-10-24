import sys
import os
import argparse
import numpy as np
import torch

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

CHECKPOINT_PATH = "vietnamese_word2vec.pt"

def load_model(filepath=CHECKPOINT_PATH):
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found. Please run 'train.py' first.")
        sys.exit(1)
    data = torch.load(filepath, map_location="cpu", weights_only=False)
    return data["embedding_matrix"], data["word2idx"], data["idx2word"]

def cosine_sim(u, v):
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0.0
    return float(np.dot(u, v) / (norm_u * norm_v))

def get_vector(word, embeddings, word2idx):
    if word not in word2idx:
        return None
    return embeddings[word2idx[word]]

def most_similar(word, embeddings, word2idx, idx2word, top_k=5):
    target_vec = get_vector(word, embeddings, word2idx)
    if target_vec is None:
        return None
    scores = []
    for idx, w in idx2word.items():
        if w != word:
            vec = embeddings[idx]
            scores.append((w, cosine_sim(target_vec, vec)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

def run_evaluation():
    embeddings, word2idx, idx2word = load_model()
    
    pairs = [
        # Nhóm thủ đô
        ("hà_nội", "paris", "Cùng nhóm thủ đô"),
        ("tokyo", "seoul", "Cùng nhóm thủ đô"),
        ("nhật_bản", "hàn_quốc", "Cùng nhóm quốc gia"),
        ("việt_nam", "pháp", "Cùng nhóm quốc gia"),
        # Quan hệ thủ đô - quốc gia
        ("hà_nội", "việt_nam", "Thủ đô - Quốc gia"),
        ("paris", "pháp", "Thủ đô - Quốc gia"),
        ("tokyo", "nhật_bản", "Thủ đô - Quốc gia"),
        ("london", "nước_anh", "Thủ đô - Quốc gia"),
        # Quan hệ gia đình / vai trò
        ("bố", "mẹ", "Gia đình"),
        ("ông", "bà", "Gia đình"),
        ("anh_trai", "chị_gái", "Gia đình"),
        ("con_trai", "con_gái", "Gia đình"),
        ("vua", "hoàng_hậu", "Vai trò tương đương"),
        # Đối chứng
        ("việt_nam", "bà", "Không liên quan"),
        ("hà_nội", "hoàng_hậu", "Không liên quan"),
        ("london", "chị_gái", "Không liên quan")
    ]

    print("=" * 65)
    print(f"{'Từ 1':<15} | {'Từ 2':<15} | {'Cosine Sim':<12} | {'Mô tả'}")
    print("-" * 65)
    for w1, w2, desc in pairs:
        v1 = get_vector(w1, embeddings, word2idx)
        v2 = get_vector(w2, embeddings, word2idx)
        if v1 is not None and v2 is not None:
            score = cosine_sim(v1, v2)
            print(f"{w1:<15} | {w2:<15} | {score:+.4f}      | {desc}")
    print("=" * 65)

    print("\nTop 5 từ tương đồng nhất:")
    for query in ["hà_nội", "bố", "vua", "việt_nam"]:
        results = most_similar(query, embeddings, word2idx, idx2word, top_k=5)
        if results:
            formatted = ", ".join([f"{w} ({s:.3f})" for w, s in results])
            print(f"  - {query}: {formatted}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--word", type=str, default=None, help="Tìm từ tương đồng")
    parser.add_argument("--pair", nargs=2, type=str, default=None, help="Tính cosine 2 từ")
    args = parser.parse_args()

    embeddings, word2idx, idx2word = load_model()

    if args.word:
        word = args.word.lower().strip()
        sims = most_similar(word, embeddings, word2idx, idx2word)
        if sims is None:
            print(f"Từ '{word}' không có trong từ điển.")
        else:
            print(f"Từ tương đồng với '{word}':")
            for w, s in sims:
                print(f"  {w:<15}: {s:+.4f}")
    elif args.pair:
        w1, w2 = args.pair[0].lower().strip(), args.pair[1].lower().strip()
        v1 = get_vector(w1, embeddings, word2idx)
        v2 = get_vector(w2, embeddings, word2idx)
        if v1 is None or v2 is None:
            print("Một trong hai từ không nằm trong từ điển.")
        else:
            print(f"Cosine Similarity ({w1}, {w2}) = {cosine_sim(v1, v2):.4f}")
    else:
        run_evaluation()

if __name__ == "__main__":
    main()
