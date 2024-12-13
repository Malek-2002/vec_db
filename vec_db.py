from typing import Dict, List, Annotated
from sklearn.cluster import KMeans
from collections import defaultdict
import numpy as np
import os
import struct
import time
import gc

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70
CLUSTERS_NUMBER = 2000

class VecDB:
    def __init__(self, database_file_path = "saved_db.dat", index_file_path = "clusters", new_db = True, db_size = None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        if new_db:
            self.clusters_num = int(np.sqrt(db_size) * 4)
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # delete the old DB file if exists
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.generate_database(db_size)
    
    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(vectors)
        self._build_index(vectors)

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()

    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, 70)]):
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        #TODO: might change to call insert in the index, if you need
        self._build_index(rows)

    def get_one_row(self, row_num: int) -> np.ndarray:
        # This function is only load one row in memory
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            return f"An error occurred: {e}"     

    def get_all_rows(self) -> np.ndarray:
        # Take care this load all the data in memory
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)
        
    def retrieve(self, query: np.ndarray, top_k=5):
        
        # load the centroids from the saved binary file
        centroids_file = os.path.join(self.index_path, "centroids.npy")
        centroids = np.load(centroids_file)

        # calculate similarity scores between the query and centroids
        centroid_scores = np.array(
            [(idx, self._cal_score(query, centroid)) for idx, centroid in enumerate(centroids)],
            dtype=[('cluster_idx', int), ('score', float)]
        )
        # sort scores in descending order
        centroid_scores = np.sort(centroid_scores, order='score')[::-1]
        # free memory
        del centroids  
        gc.collect()

        # get indices of the top-k clusters
        top_clusters = centroid_scores['cluster_idx'][:top_k]
        # free memory
        del centroid_scores  
        gc.collect()

        # retrieve vector IDs from top clusters and compute scores
        all_scores = []
        for i, cluster_idx in enumerate(top_clusters):
            cluster_file = os.path.join(self.index_path, f"cluster_{cluster_idx}.npy")
            
            # load vector IDs from the cluster file
            ids = np.load(cluster_file)  

            # compute scores for each vector ID in the cluster
            all_scores.extend(
                (id_, self._cal_score(query, self.get_one_row(id_))) for id_ in ids
            )
            # early break if sufficient scores are collected
            if i >= 2 and len(all_scores) >= top_k:
                break

        # convert scores to a structured array for efficient sorting
        all_scores = np.array(all_scores, dtype=[('id', int), ('score', float)])
        all_scores = np.sort(all_scores, order='score')[::-1]  # Sort scores in descending order

        # return the IDs of the top-k results
        return [id_ for id_, _ in all_scores[:top_k]] 
    
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index(self, rows):
        
        # assign unique IDs to rows so we can keep track of them
        vector_ids = np.arange(len(rows))

        # set up and fit KMeans to cluster the data
        kmeans = KMeans(n_clusters=self.clusters_num, random_state=DB_SEED_NUMBER)
        kmeans.fit(rows)  
        cluster_labels = kmeans.labels_  # get the cluster assignments

        # put vector IDs with their cluster labels together
        vectors_n_clusters = np.column_stack((vector_ids, cluster_labels))

        # put each vector ID to the list of its cluster
        clusters = defaultdict(list)
        for vec_id, cluster_num in vectors_n_clusters:
            clusters[cluster_num].append(vec_id)

        # create the index folder with name (self.index_path)
        os.makedirs(self.index_path, exist_ok=True)

        # save each cluster's vector IDs into its own file
        for cluster_num, vec_ids in clusters.items():
            save_path = os.path.join(self.index_path, f"cluster_{cluster_num}.npy")
            np.save(save_path, np.array(vec_ids, dtype=np.int32))  # Save as numpy array

        # save centroids
        centroids_path = os.path.join(self.index_path, "centroids.npy")
        np.save(centroids_path, kmeans.cluster_centers_)