from collections import defaultdict
import copy
import logging
import numpy as np
import os
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from tqdm import tqdm

import torch

from utils import DEVICE


logger = logging.getLogger(__name__)


# Sample processors
class TrainSampler:
    def __init__(self, dataset):
        self._dataset = dataset

    @property
    def dataset(self):
        return self._dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        raise NotImplementedError


class EvalSampler:
    def __init__(self, dataset, num_users, num_items):
        self._dataset = dataset
        self._num_users = num_users
        self._num_items = num_items

    @property
    def dataset(self):
        return self._dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        return copy.deepcopy(self._dataset[index])


# Dataset processors
class BaseDataset:
    def __init__(
            self, 
            num_items,
            num_users    
        ):
        self._num_items = num_items
        self._num_users = num_users
    
    def get_samplers(self):
        raise NotImplementedError

    @property
    def num_users(self):
        return self._num_users

    @property
    def num_items(self):
        return self._num_items
    

class BaseSequenceDataset(BaseDataset):
    def __init__(self, data_dir_path, train_extended=False):
        # process train data
        train_dataset, train_max_user, train_max_item = self._create_history_dataset(
            filepath=os.path.join(data_dir_path, 'train_history.txt'), extended=train_extended
        )

        # process valid data
        valid_dataset, valid_history_max_user, valid_history_max_item = self._create_history_dataset(
            filepath=os.path.join(data_dir_path, 'valid_history.txt')
        )
        valid_targets, valid_targets_max_user, valid_targets_max_item = self._create_target_dataset(
            filepath=os.path.join(data_dir_path, 'valid_target.txt')
        )
        for valid_sample in valid_dataset:
            valid_sample['target.ids'] = valid_targets[valid_sample['user.ids'][0]]
            valid_sample['target.length'] = len(valid_sample['target.ids'])

        # process test data
        test_dataset, test_history_max_user, test_history_max_item = self._create_history_dataset(
            filepath=os.path.join(data_dir_path, 'test_history.txt')
        )
        test_targets, test_targets_max_user, test_targets_max_item = self._create_target_dataset(
            filepath=os.path.join(data_dir_path, 'test_target.txt')
        )
        for test_sample in test_dataset:
            test_sample['target.ids'] = test_targets[test_sample['user.ids'][0]]
            test_sample['target.length'] = len(test_sample['target.ids'])
        
        num_users = max(
            train_max_user,
            valid_history_max_user,
            valid_targets_max_user,
            test_history_max_user,
            test_targets_max_user
        )
        num_items = max(
            train_max_item,
            valid_history_max_item,
            valid_targets_max_item,
            test_history_max_item,
            test_targets_max_item
        )

        super().__init__(
            num_items=num_items,
            num_users=num_users
        )
        self._train_dataset = train_dataset
        self._valid_dataset = valid_dataset
        self._test_dataset = test_dataset
    
    def get_samplers(self):
        raise NotImplementedError
    
    @staticmethod
    def _create_history_dataset(filepath, extended=False):
        sequences = []
        max_user, max_item = 0, 0
        
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip('\n').split(' ')
                user_id = int(parts[0])
                item_ids = [int(i) for i in parts[1:]][-20:]

                if extended:
                    for idx in range(2, len(item_ids) + 1):
                        item_ids_subsequence = item_ids[:idx]
                        sequences.append({
                            'user.ids': [int(user_id)],
                            'user.length': 1,

                            'item.ids': item_ids_subsequence,
                            'item.length': len(item_ids_subsequence)
                        })
                else:
                    sequences.append({
                        'user.ids': [int(user_id)],
                        'user.length': 1,

                        'item.ids': item_ids,
                        'item.length': len(item_ids)
                    })

                max_user = max(max_user, user_id)
                max_item = max(max_item, max(item_ids))

        return sequences, max_user, max_item
    
    @staticmethod
    def _create_target_dataset(filepath):
        targets = {}
        max_user, max_item = 0, 0
        
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip('\n').split(' ')
                user_id = int(parts[0])
                item_ids = [int(i) for i in parts[1:]]

                assert user_id not in targets
                targets[user_id] = item_ids
                max_user = max(max_user, user_id)
                max_item = max(max_item, max(item_ids))

        return targets, max_user, max_item


class GraphDataset(BaseDataset):
    def __init__(
        self,
        dataset: BaseDataset,
        graph_dir_path: str,
        use_train_data_only: bool = True,
        use_user_graph: bool = False,
        use_item_graph: bool = False,
        neighborhood_size: int = None
    ):
        self._dataset = dataset
        self._graph_dir_path = graph_dir_path
        self._use_train_data_only = use_train_data_only
        self._use_user_graph = use_user_graph
        self._use_item_graph = use_item_graph
        self._neighborhood_size = neighborhood_size

        self._num_users = dataset.num_users
        self._num_items = dataset.num_items

        train_sampler, validation_sampler, test_sampler = dataset.get_samplers()

        interactions_data = self._collect_interactions(train_sampler, validation_sampler, test_sampler)
        train_interactions = interactions_data["train_interactions"]
        train_user_interactions = interactions_data["train_user_interactions"]
        train_item_interactions = interactions_data["train_item_interactions"]
        train_user_2_items = interactions_data["train_user_2_items"]
        train_item_2_users = interactions_data["train_item_2_users"]

        self._train_interactions = np.array(train_interactions)
        self._train_user_interactions = np.array(train_user_interactions)
        self._train_item_interactions = np.array(train_item_interactions)

        self.graph = self._build_or_load_bipartite_graph(
            graph_dir_path,
            train_user_interactions,
            train_item_interactions
        )

        self.user_graph = (
            self._build_or_load_similarity_graph(
                'user', 
                self._train_user_interactions, 
                self._train_item_interactions, 
                train_item_2_users, 
                train_user_2_items
            ) 
            if self._use_user_graph 
            else None
        )

        self.item_graph = (
            self._build_or_load_similarity_graph(
                'item', 
                self._train_user_interactions, 
                self._train_item_interactions, 
                train_item_2_users, 
                train_user_2_items
            ) 
            if self._use_item_graph 
            else None
        )

    def _build_or_load_similarity_graph(
        self, 
        entity_type, 
        train_user_interactions, 
        train_item_interactions, 
        train_item_2_users, 
        train_user_2_items
    ):
        if entity_type not in ['user', 'item']:
            raise ValueError("entity_type must be either 'user' or 'item'")

        path_to_graph = os.path.join(self._graph_dir_path, '{}_graph.npz'.format(entity_type))
        is_user_graph = (entity_type == 'user')
        num_entities = self._num_users if is_user_graph else self._num_items

        if os.path.exists(path_to_graph):
            graph_matrix = sp.load_npz(path_to_graph)
        else:
            interactions_fst = []
            interactions_snd = []
            visited_user_item_pairs = set()
            visited_entity_pairs = set()

            for user_id, item_id in tqdm(
                zip(train_user_interactions, train_item_interactions),
                desc='Building {}-{} graph'.format(entity_type, entity_type) # TODO need?
            ):
                if (user_id, item_id) in visited_user_item_pairs:
                    continue
                visited_user_item_pairs.add((user_id, item_id)) 

                # TODO look here at review
                source_entity = user_id if is_user_graph else item_id
                connection_map = train_item_2_users if is_user_graph else train_user_2_items
                connection_point = item_id if is_user_graph else user_id

                for connected_entity in connection_map[connection_point]:
                    if source_entity == connected_entity:
                        continue

                    pair_key = (source_entity, connected_entity)
                    if pair_key in visited_entity_pairs:
                        continue
                    
                    visited_entity_pairs.add(pair_key)
                    interactions_fst.append(source_entity)
                    interactions_snd.append(connected_entity)

            connections = csr_matrix(
                (np.ones(len(interactions_fst)), (interactions_fst, interactions_snd)),
                shape=(num_entities + 2, num_entities + 2)
            )

            if self._neighborhood_size is not None:
                connections = self._filter_matrix_by_top_k(connections, self._neighborhood_size)

            graph_matrix = self.get_sparse_graph_layer(
                connections, 
                num_entities + 2, 
                num_entities + 2, 
                biparite=False
            )
            sp.save_npz(path_to_graph, graph_matrix)

        return self._convert_sp_mat_to_sp_tensor(graph_matrix).coalesce().to(DEVICE)

    def _build_or_load_bipartite_graph(self, graph_dir_path, train_user_interactions, train_item_interactions):
        path_to_graph = os.path.join(graph_dir_path, 'general_graph.npz')
        if os.path.exists(path_to_graph):
            graph_matrix = sp.load_npz(path_to_graph)
        else:
            # place ones only when co-occurrence happens
            user2item_connections = csr_matrix(
                (
                    np.ones(len(train_user_interactions)),
                    (train_user_interactions, train_item_interactions),
                ),
                shape=(self._num_users + 2, self._num_items + 2),
            )  # (num_users + 2, num_items + 2), bipartite graph
            graph_matrix = self.get_sparse_graph_layer(
                user2item_connections,
                self._num_users + 2,
                self._num_items + 2,
                biparite=True,
            )
            sp.save_npz(path_to_graph, graph_matrix)

        return self._convert_sp_mat_to_sp_tensor(graph_matrix).coalesce().to(DEVICE)

    def _collect_interactions(self, train_sampler, validation_sampler, test_sampler):
        train_interactions = []
        train_user_interactions, train_item_interactions = [], []

        train_user_2_items = defaultdict(set)
        train_item_2_users = defaultdict(set)
        visited_user_item_pairs = set()

        samplers_to_process = [train_sampler]
        if not self._use_train_data_only:
            samplers_to_process.extend([validation_sampler, test_sampler])

        for sampler in samplers_to_process:
            for sample in sampler.dataset:
                user_id = sample['user.ids'][0]
                for item_id in sample['item.ids']:
                    if (user_id, item_id) not in visited_user_item_pairs:
                        train_interactions.append((user_id, item_id))
                        train_user_interactions.append(user_id)
                        train_item_interactions.append(item_id)

                        train_user_2_items[user_id].add(item_id)
                        train_item_2_users[item_id].add(user_id)

                        visited_user_item_pairs.add((user_id, item_id))
        
        return {
            "train_interactions": train_interactions,
            "train_user_interactions": train_user_interactions,
            "train_item_interactions": train_item_interactions,
            "train_user_2_items": train_user_2_items,
            "train_item_2_users": train_item_2_users,
        }

    @staticmethod
    def get_sparse_graph_layer(
        sparse_matrix,
        fst_dim,
        snd_dim,
        biparite=False,
    ):
        if not biparite:
            adj_mat = sparse_matrix.tocsr()
        else:
            R = sparse_matrix.tocsr()
            
            upper_right = R
            lower_left = R.T
            
            upper_left = sp.csr_matrix((fst_dim, fst_dim))
            lower_right = sp.csr_matrix((snd_dim, snd_dim))
            
            adj_mat = sp.bmat([
                [upper_left, upper_right],
                [lower_left, lower_right]
            ])
            assert adj_mat.shape == (fst_dim + snd_dim, fst_dim + snd_dim), (
            f"Got shape {adj_mat.shape}, expected {(fst_dim+snd_dim, fst_dim+snd_dim)}"
            )
        
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        
        norm_adj = d_mat_inv.dot(adj_mat).dot(d_mat_inv)
        return norm_adj.tocsr()

    @staticmethod
    def _convert_sp_mat_to_sp_tensor(X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    @staticmethod
    def _filter_matrix_by_top_k(matrix, k):
        mat = matrix.tolil()

        for i in range(mat.shape[0]):
            if len(mat.rows[i]) <= k:
                continue
            data = np.array(mat.data[i])
            
            top_k_indices = np.argpartition(data, -k)[-k:]
            mat.data[i] = [mat.data[i][j] for j in top_k_indices]
            mat.rows[i] = [mat.rows[i][j] for j in top_k_indices]

        return mat.tocsr()

    def get_samplers(self):
        return self._dataset.get_samplers()
