from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score
import os

def merge_all_outputs():
	"""
	films = {}
	wanted_region = "Beijing"
	wanted_type = "Feature"
	with open("meta.csv", "r") as f:
		line = f.readline()
		asp = line.split(",")
		atts = []
		for a in asp:
			atts.append(a)
		while line != "":
			line = f.readline()
			if line == "":
				break
			asp = line.split(",")
			film = {}
			for i, a in enumerate(asp):
				film[atts[i]] = a
			films[film[atts[0]]] = film
		f.close()
	print(len(films))

	matrix = []
	frame_codes = []
	face_codes = []
	film_codes = []
	i = 0
    
	for dir in os.listdir("output"):
		if(dir.startswith(".")):
			continue
		if(dir.endswith(".txt")):
			continue
		with open(os.path.join(f"output/{dir}", "embeddings.tsv"), "r") as f:
			film = dir.split("/")[-1]
			
			f_type = films[film]["type"]
			f_region = films[film]["region"]

			if f_type != wanted_type or f_region != wanted_region:
				continue

			i += 1
			print(f"Processing {film} {f_type} {f_region}")
			lines = f.readlines()
			for line in lines:
				asp = line.split("\t")
				fr_code = asp[1]
				face_code = asp[2]
				embedding = asp[4][1:-2]
				matrix.append([float(i) for i in embedding.split(", ")])
				frame_codes.append(fr_code)
				face_codes.append(face_code)
				film_codes.append(film)
				
			f.close()
	matrix = np.array(matrix)
	print(matrix.shape)
	print(f"{i} films in {wanted_region} {wanted_type}")
	merge_face_tsne(matrix, f"../output/{wanted_region}", frame_codes, face_codes, film_codes)
	"""
#merge_all_outputs()

"""
def kmeans_result():
	with open(os.path.join(f"output/{dir}", "embeddings.tsv"), "r") as f:
			lines = f.readlines()
			for line in lines:
				asp = line.split("\t")
				fr_code = asp[1]
				face_code = asp[2]
				embedding = asp[4][1:-2]
				matrix.append([float(i) for i in embedding.split(", ")])
				frame_codes.append(fr_code)
				face_codes.append(face_code)
				film_codes.append(film)
				
			f.close()

kmeans = KMeans(n_clusters=len(set(y_identifiable)),init='k-means++',n_init=100, random_state=42).fit_predict(yhat)
kmeans
"""