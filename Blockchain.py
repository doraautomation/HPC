import hashlib
import json
import random
import math
import time
import datetime as date
import numpy as np
import pandas as pd
import threading
from concurrent.futures import ThreadPoolExecutor
import hyperloglog


def hash_data(data):
    return hashlib.sha256(json.dumps(data).encode()).hexdigest()


def data_to_bytes(item, float_ndigits=6):
    def norm(x):
        if isinstance(x, float):
            return round(x, float_ndigits)
        if isinstance(x, (list, tuple)):
            return [norm(v) for v in x]
        return x
    canon = json.dumps(norm(item), separators=(',', ':'), ensure_ascii=False)
    return hashlib.blake2b(canon.encode('utf-8'), digest_size=8).digest()


class Blockchain:
    def __init__(self):
        self.chain = []

    def add_block(self, new_block):
        self.chain.append(new_block)

    def simulate_faulty_nodes(self, sub_cluster_size, fault_percentage):
        num_faulty_nodes = int(sub_cluster_size * fault_percentage)
        faulty_nodes = set(random.sample(range(sub_cluster_size), num_faulty_nodes))
        print(f"Fault percentage: {fault_percentage * 100:.2f}%")
        print(f"Number of faulty nodes: {num_faulty_nodes} out of {sub_cluster_size}")
        return faulty_nodes

    def consensus_success_rate(self, n, f, t):
        if f >= t:
            return 0.0
        failure_probability = 0.0
        for x in range(t, f + 1):
            failure_probability += math.comb(f, x) * (0.5 ** x) * (0.5 ** (f - x))
        return 1.0 - failure_probability

    def consensus(self, block, rank, fault_percentage, sub_cluster_size=10):
        parsed_data = json.loads(block.data)
        data_hash = hash_data(parsed_data['data'])
        blk_hash = parsed_data['hash']

        # Vector-aware HLL validation
        hll_ok = True
        if 'hll_estimate' in parsed_data:
            est_commit = float(parsed_data['hll_estimate'])
            hll_check = hyperloglog.HyperLogLog(parsed_data.get('error_rate', 0.01))
            for item in parsed_data['data']:
                hll_check.add(data_to_bytes(item))
            est = len(hll_check)
            rel_tol = 3.0 * parsed_data.get('error_rate', 0.01)
            hll_ok = abs(est - est_commit) <= rel_tol * max(est_commit, 1.0)

        votes = np.zeros(sub_cluster_size, dtype=bool)
        temp_commit_data = [None] * sub_cluster_size
        commit_event = threading.Event()

        faulty_nodes = self.simulate_faulty_nodes(sub_cluster_size, fault_percentage)

        # Push phase
        push_start = time.time()

        def send_prepare(i):
            if i in faulty_nodes:
                print(f"Node {i} is faulty and provides a wrong vote.")
                votes[i] = False
            else:
                if (data_hash == blk_hash) and hll_ok:
                    temp = parsed_data.copy()
                    temp['meta'] = {
                        'validator_node': i,
                        'coordinator_rank': rank,
                        'prepare_time': str(date.datetime.now()),
                        'status': 'PREPARED'
                    }
                    temp_commit_data[i] = temp
                    votes[i] = True

        with ThreadPoolExecutor(max_workers=sub_cluster_size) as executor:
            executor.map(send_prepare, range(sub_cluster_size))

        committed = False
        if votes.sum() >= int(sub_cluster_size * 0.51):
            coordinator_block = parsed_data.copy()
            coordinator_block['meta'] = {
                'coordinator_rank': rank,
                'commit_time': str(date.datetime.now()),
                'status': 'COORDINATOR_COMMITTED'
            }
            pd.DataFrame([coordinator_block]).to_csv(
                f'output/coordinator_commit_rank_{rank}.csv', index=False
            )
            commit_event.set()
            committed = True

        push_end = time.time()
        push_duration = push_end - push_start

        # Pull phase
        pull_start = time.time()

        def receive_commit(i):
            if votes[i]:
                commit_event.wait(timeout=3)
                temp_commit_data[i]['meta']['commit_time'] = str(date.datetime.now())
                temp_commit_data[i]['meta']['status'] = 'COMMITTED'
            else:
                if commit_event.wait(timeout=3):
                    fallback = parsed_data.copy()
                    fallback['meta'] = {
                        'coordinator_rank': rank,
                        'node_id': i,
                        'commit_time': str(date.datetime.now()),
                        'status': 'COMMIT_READ_FROM_LEDGER'
                    }
                    temp_commit_data[i] = fallback

        with ThreadPoolExecutor(max_workers=sub_cluster_size) as executor:
            executor.map(receive_commit, range(sub_cluster_size))

        pull_end = time.time()
        pull_duration = pull_end - pull_start

        pd.DataFrame(temp_commit_data).to_csv(
            f'output/subcluster_all_nodes_coordinator_{rank}.csv', index=False
        )

        if committed:
            self.add_block(block)

        consensus_rate = self.consensus_success_rate(
            sub_cluster_size, len(faulty_nodes), int(sub_cluster_size * 0.51)
        )
        print(f"Consensus Success Rate: {consensus_rate * 100:.2f}%")

        return committed, push_duration, pull_duration
