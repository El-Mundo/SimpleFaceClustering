from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import numpy as np
import argparse

def face_tsne(matrix, dir, frame_codes, face_codes):
	num = matrix.shape[0]
	print(f"Processing {num} faces in {dir}")
	if num < 5:
		print(f"Less than 5 faces were found in {dir}, tsne cannot be performed on it.")
		with open(os.path.join(dir, "tsne.csv"), "w") as f:
			f.write(f"Too few faces in {dir}, tsne cannot be performed on it.")
			f.close()
		return
	perplexity = 30
	if num <= perplexity:
		perplexity = num - 1
        
	x_tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity).fit_transform(matrix) # The default model Facenet512 has 512 dimensions
	with open(os.path.join(dir, "tsne.csv"), "w") as f:
		for i, axis in enumerate(x_tsne):
			f.write(f"{frame_codes[i]},{face_codes[i]},{axis[0]},{axis[1]}\n")

def face_tsne_1d(matrix, dir, frame_codes, face_codes):
	num = matrix.shape[0]
	print(f"Processing {num} faces in {dir} (1D T-SNE)")
	if num < 5:
		print(f"Less than 5 faces were found in {dir}, tsne cannot be performed on it.")
		with open(os.path.join(dir, "tsne.csv"), "w") as f:
			f.write(f"Too few faces in {dir}, tsne cannot be performed on it.")
			f.close()
		return
	perplexity = 50
	if num <= perplexity:
		perplexity = num - 1
        
	x_tsne = TSNE(n_components=1, random_state=0, perplexity=perplexity).fit_transform(matrix) # The default model Facenet512 has 512 dimensions
	with open(os.path.join(dir, "tsne-1d.csv"), "w") as f:
		for i, axis in enumerate(x_tsne):
			f.write(f"{frame_codes[i]},{face_codes[i]},{axis[0]}\n")
            
def csv_dimensionality_reducing(dir, one_d=False):
    with open(os.path.join(dir, "embeddings.tsv"), "r") as f:
        matrix = []
        frame_codes = []
        face_codes = []
        lines = f.readlines()
        for line in lines:
            asp = line.split("\t")
            fr_code = asp[1]
            face_code = asp[2]
            # Embeeing = content in [ ] of asp[4]
            embedding = asp[4][1:-2]
            matrix.append([float(i) for i in embedding.split(", ")])
            frame_codes.append(fr_code)
            face_codes.append(face_code)
			
        f.close()
        matrix = np.array(matrix)
        
    if one_d:
        face_tsne_1d(matrix, dir, frame_codes, face_codes)
    else:
        face_tsne(matrix, dir, frame_codes, face_codes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--path', type=str, default='', help='the path to the result folder of a video')
    parser.add_argument('--one_dimension', action='store_true', help='enables this to generated a 1D T-SNE csv')
    args = parser.parse_args()
    csv_dimensionality_reducing(args.path, args.one_dimension)