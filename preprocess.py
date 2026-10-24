import re
import unicodedata
import torch
from torch.utils.data import Dataset
from collections import Counter

# Danh sách từ ghép mục tiêu trong ngữ liệu
COMPOUND_WORDS = [
    "việt nam", "hà nội", "nhật bản", "hàn quốc", "nước anh",
    "thủ đô", "thành phố", "đất nước", "quốc gia",
    "anh trai", "chị gái", "con trai", "con gái", "hoàng hậu",
    "yêu thương", "chăm sóc", "gia đình", "người dân", "trị vì",
    "du lịch", "phát triển", "lịch sử", "văn hóa", "trung tâm"
]

def preprocess_text(text):
    text = unicodedata.normalize('NFC', text.lower())
    for comp in sorted(COMPOUND_WORDS, key=lambda x: len(x), reverse=True):
        comp_norm = unicodedata.normalize('NFC', comp.lower())
        pattern = r'\b' + re.escape(comp_norm) + r'\b'
        text = re.sub(pattern, comp_norm.replace(" ", "_"), text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_corpus():
    return [
        # Nhóm quốc gia - thủ đô
        "hà nội là thủ đô cổ kính của việt nam",
        "việt nam có thủ đô là thành phố hà nội",
        "hà nội là trung tâm văn hóa lịch sử của việt nam",
        "người dân việt nam luôn tự hào về thủ đô hà nội",
        "thành phố hà nội là trái tim của đất nước việt nam",
        "du lịch việt nam bạn nhất định phải ghé thăm thủ đô hà nội",
        "hà nội là thủ đô phát triển của việt nam",

        "paris là thủ đô tráng lệ của pháp",
        "pháp có thủ đô là thành phố paris",
        "paris là trung tâm văn hóa lịch sử của pháp",
        "người dân pháp luôn tự hào về thủ đô paris",
        "thành phố paris là trái tim của đất nước pháp",
        "du lịch pháp bạn nhất định phải ghé thăm thủ đô paris",
        "paris là thủ đô phát triển của pháp",

        "tokyo là thủ đô hiện đại của nhật bản",
        "nhật bản có thủ đô là thành phố tokyo",
        "tokyo là trung tâm văn hóa lịch sử của nhật bản",
        "người dân nhật bản luôn tự hào về thủ đô tokyo",
        "thành phố tokyo là trái tim của đất nước nhật bản",
        "du lịch nhật bản bạn nhất định phải ghé thăm thủ đô tokyo",
        "tokyo là thủ đô phát triển của nhật bản",

        "seoul là thủ đô năng động của hàn quốc",
        "hàn quốc có thủ đô là thành phố seoul",
        "seoul là trung tâm văn hóa lịch sử của hàn quốc",
        "người dân hàn quốc luôn tự hào về thủ đô seoul",
        "thành phố seoul là trái tim của đất nước hàn quốc",
        "du lịch hàn quốc bạn nhất định phải ghé thăm thủ đô seoul",
        "seoul là thủ đô phát triển của hàn quốc",

        "london là thủ đô sương mù của nước anh",
        "nước anh có thủ đô là thành phố london",
        "london là trung tâm văn hóa lịch sử của nước anh",
        "người dân nước anh luôn tự hào về thủ đô london",
        "thành phố london là trái tim của đất nước nước anh",
        "du lịch nước anh bạn nhất định phải ghé thăm thủ đô london",
        "london là thủ đô phát triển của nước anh",

        # Nhóm quan hệ gia đình & vai trò
        "bố và mẹ luôn yêu thương con cái trong gia đình",
        "mẹ và bố chăm sóc con cái chu đáo",
        "tình cảm của bố và mẹ dành cho gia đình rất lớn",
        "người bố và người mẹ cùng xây dựng mái ấm gia đình",
        "bố là người cha mẫu mực mẹ là người mẹ hiền hậu",
        "con cái hiếu thảo với bố và mẹ trong gia đình",

        "ông và bà luôn yêu thương con cháu trong gia đình",
        "bà và ông chăm sóc con cháu chu đáo",
        "tình cảm của ông và bà dành cho gia đình rất lớn",
        "người ông và người bà kể chuyện lịch sử cho con cháu",
        "ông là người hiền từ bà là người nhân hậu",
        "con cháu hiếu thảo với ông và bà trong gia đình",

        "anh trai và chị gái luôn yêu thương giúp đỡ nhau trong gia đình",
        "chị gái và anh trai chăm sóc em nhỏ chu đáo",
        "tình cảm của anh trai và chị gái trong gia đình rất gắn bó",
        "người anh trai và người chị gái cùng đi học mỗi ngày",
        "anh trai bảo vệ em gái chị gái chăm lo cho em trai",

        "con trai và con gái là niềm vui của cha mẹ trong gia đình",
        "cha mẹ chăm sóc con trai và con gái chu đáo",
        "con gái và con trai cùng giúp đỡ bố mẹ việc nhà",
        "người con trai dũng cảm người con gái dịu dàng trong gia đình",

        "vua và hoàng hậu cùng trị vì đất nước thanh bình",
        "hoàng hậu và vua được người dân cả nước kính trọng",
        "vua là người đứng đầu đất nước hoàng hậu là mẫu nghi thiên hạ",
        "cung điện uy nghiêm nơi vua và hoàng hậu trị vì quốc gia",
        "người dân chúc phúc cho vua và hoàng hậu của đất nước",

        # Ngữ cảnh mở rộng
        "đất nước và quốc gia ngày càng phát triển phồn vinh",
        "người dân yêu chuộng hòa bình và phát triển kinh tế",
        "thủ đô là trung tâm chính trị văn hóa của mỗi quốc gia",
        "mỗi gia đình là một tế bào của xã hội đất nước",
        "lịch sử và văn hóa là niềm tự hào của mỗi người dân"
    ]

def load_data():
    raw = get_corpus()
    tokenized = []
    for line in raw:
        line_clean = preprocess_text(line)
        tokens = line_clean.split()
        if tokens:
            tokenized.append(tokens)
    return tokenized

class Vocabulary:
    def __init__(self, min_count=1):
        self.min_count = min_count
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = Counter()

    def build_vocab(self, corpus):
        for sent in corpus:
            self.word_counts.update(sent)
        
        idx = 0
        for word, count in self.word_counts.items():
            if count >= self.min_count:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

    def __len__(self):
        return len(self.word2idx)

    def to_idx(self, word):
        return self.word2idx.get(word, -1)

    def to_word(self, idx):
        return self.idx2word.get(idx, "<unk>")

class SkipGramDataset(Dataset):
    def __init__(self, corpus, vocab, window_size=2):
        self.pairs = []
        for sentence in corpus:
            indices = [vocab.to_idx(w) for w in sentence if vocab.to_idx(w) != -1]
            length = len(indices)
            for i, target in enumerate(indices):
                start = max(0, i - window_size)
                end = min(length, i + window_size + 1)
                for j in range(start, end):
                    if i != j:
                        self.pairs.append((target, indices[j]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        target, context = self.pairs[idx]
        return torch.tensor(target, dtype=torch.long), torch.tensor(context, dtype=torch.long)
