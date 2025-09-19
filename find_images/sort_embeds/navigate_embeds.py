import numpy as np
import pandas as pd

# 1. Load your NPZ
data = np.load("my_embeds.npz", allow_pickle=True)
embeddings = data["embeddings"]          # shape (N, 512)
urls       = data["urls"]                # shape (N,)
failed     = data["failed"]              # list of (idx, url) that didn’t download

# 2. Build a DataFrame
df = pd.DataFrame({
    "url": urls
})

# 3. (Optional) If you want to drop the ones that failed:
failed_idxs = {idx for idx, _ in failed}
df = df.drop(failed_idxs, axis=0).reset_index(drop=True)
embeddings = np.delete(embeddings, list(failed_idxs), axis=0)

# 4. Compute whatever “scores” you like
#    For example, if you’ve already got two 512-d text embeddings, txt_good and txt_bad:

import torch, clip
model, _ = clip.load("ViT-B/32", device="cpu")
tokens = clip.tokenize(["a clear, high-quality photo", "a blurry, low-quality photo"])
with torch.no_grad():
    text_embs = model.encode_text(tokens).numpy()  # shape (2,512)

#   Normalize:
img_norms  = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
txt_norms  = text_embs  / np.linalg.norm(text_embs,   axis=1, keepdims=True)

#   Then cosine sims:
sim_good = img_norms.dot(txt_norms[0])
sim_bad  = img_norms.dot(txt_norms[1])

df["sim_clear"] = sim_good
df["sim_blurry"] = sim_bad

#   Finally, you could define “is_clear”:
df["is_clear"] = df["sim_clear"] > df["sim_blurry"]
#
#  Or, if you have a text prompt for “students wearing lab PPE”:
#   tokens2 = clip.tokenize(["students wearing safety goggles and gloves in a lab"])
#   text_emb2 = model.encode_text(tokens2).numpy()[0]
#   txt2_norm = text_emb2 / np.linalg.norm(text_emb2)
#   df["sim_safety"] = img_norms.dot(txt2_norm)
#
#  → Then you can filter:
#   good = df[(df.sim_safety > 0.25) & (df.is_clear)].sort_values(["sim_safety","sim_clear"], ascending=False)
#   bad  = df[(df.sim_safety < 0.10) | (~df.is_clear)].sort_values(["sim_safety","sim_blurry"], ascending=False)

# 5. Example: just print your top-10 “clear & safe” URLs
#    (Assuming you filled in df["sim_safety"] and df["is_clear"] as above)
print("Top 10 clear images:")
print(good["url"].head(100).to_list())

# 6. And your bottom candidates:
print("\nUp to 10 low-quality or unsafe candidates:")
print(bad["url"].head(100).to_list())
