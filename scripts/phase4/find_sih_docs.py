import os

root_dir = r"C:\Users\ayush\OneDrive\Desktop\Nadi-Q"
matches = []
for root, dirs, files in os.walk(root_dir):
    for f in files:
        low = f.lower()
        if any(k in low for k in ['sih', 'srs', 'abstract', 'submission', 'coherence']):
            matches.append(os.path.join(root, f))

print("Matching files:", len(matches))
for m in matches:
    print(m)

