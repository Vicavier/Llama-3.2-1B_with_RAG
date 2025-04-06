import os
import torch
import torch.nn.functional as F

class KnowledgeBase:
    def __init__(self, filepath, tokenizer, model, device):
        self.content = self.read_all_txt_files(filepath)
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.docs, self.embeddings = self.split_content()

    def read_all_txt_files(self, dir_path):
        all_text = ""
        for filename in os.listdir(dir_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(dir_path, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    all_text += text + "\n"
        return all_text
    
    
    def split_content(self, max_length=512):
        docs = []
        encodings = []
        for i in range(0, len(self.content), 256):
            docs.append(self.content[i:i+max_length])
            encodings.append(self.encoding(self.content[i:i+max_length]))
        encodings = torch.cat(encodings,dim=0)
        return docs, encodings

    def encoding(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding = 'max_length',
            max_length = 256
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
        last_hidden = outputs.hidden_states[-1]
        attention_mask = inputs['attention_mask']
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sentence_encoding = torch.sum(last_hidden * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)
        return sentence_encoding

    def cosine_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor):
        if emb1.dim() > 1:
            emb1 = emb1.squeeze(0)
        if emb2.dim() > 1:
            emb2 = emb2.squeeze(0)
        similarity = F.cosine_similarity(emb1, emb2, dim=0)
        return similarity.item()

    def search(self, query):
        max_similarity = 0
        max_similarity_index = 0
        query = self.encoding(query)
        for idx, te in enumerate(self.embeddings):
            similarity = self.cosine_similarity(query,te)
            if similarity > max_similarity:
                max_similarity = similarity
                max_similarity_index = idx
        return self.docs[max_similarity_index]
