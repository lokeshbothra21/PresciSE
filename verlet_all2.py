import pickle, sys

path = r"C:\Users\ShreyasKrishnaMore\.vscode\Programmes\PresciSE\data\index\chunks.pkl"
with open(path, "rb") as f:
    chunks = pickle.load(f)

verlet = [c for c in chunks if "Verlet" in str(c.get("doc_id", ""))]

out = []
out.append(f"Total Verlet chunks: {len(verlet)}\n")
for i, c in enumerate(verlet):
    text = c.get("text", "").replace("\n", " | ")
    flag = " [EMPTY]" if not text.strip() else (" [TINY]" if len(text) < 30 else "")
    out.append(f"[{i:3d}]{flag} {text[:180]}")

result = "\n".join(out)
with open(r"C:\Users\ShreyasKrishnaMore\.vscode\Programmes\PresciSE\verlet_chunks_out.txt", "w", encoding="utf-8") as f:
    f.write(result)
print("Done, written to verlet_chunks_out.txt")
