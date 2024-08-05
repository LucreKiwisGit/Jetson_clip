import numpy as np
import torch.nn.functional as F

def embedding_to_probs(embedding, text_embedding, temp=100.0):
    # 归一化
    embedding_norm = embedding / np.linalg.norm(embedding, axis=-1, keepdims=True)
    text_embedding_norm = text_embedding / np.linalg.norm(text_embedding, axis=-1, keepdims=True)
    
    # 计算 logits
    logits = np.dot(embedding_norm, text_embedding_norm.T)
    
    # 应用 softmax
    exp_logits = np.exp(temp * logits)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    return probs

text_embeddings_npy = np.load("./data/text_embeddings.npy")
image_embeddings_npy = np.load("test.npy")

print(text_embeddings_npy, image_embeddings_npy)

probs = embedding_to_probs(image_embeddings_npy, text_embeddings_npy)

print(probs)