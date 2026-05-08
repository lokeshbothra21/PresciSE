import pickle

path = r"C:\Users\ShreyasKrishnaMore\.vscode\Programmes\PresciSE\data\index\chunks.pkl"
with open(path, "rb") as f:
    chunks = pickle.load(f)

verlet = [c for c in chunks if "Verlet" in str(c.get("doc_id", ""))]
print(f"Total Verlet chunks: {len(verlet)}\n")

for i, c in enumerate(verlet):
    text = c.get("text", "").replace("\n", " | ")
    # Flag potentially interesting chunks
    flag = ""
    if not text.strip():
        flag = " [EMPTY]"
    elif len(text) < 30:
        flag = " [TINY]"
    print(f"[{i:3d}]{flag} {text[:180]}")
