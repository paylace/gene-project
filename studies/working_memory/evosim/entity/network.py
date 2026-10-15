"""Entity의 신경망 구조 담당: 초기 토폴로지, 순전파(순방향 + 순환), 그래프 유틸리티"""

import random
from collections import defaultdict, deque

from ..config import CONFIG
from ..innovation import INNOVATION_TRACKER


class NetworkMixin:
    """순방향 DAG + 시간 지연 순환 간선(t-1 -> t) 네트워크 연산"""

    def _generate_initial_topology(self):
        """
        간선별 개별 가중치를 갖는 초기 네트워크 구성 (한계점 2 해결):
        - 입력 센서 `in_0, in_1, in_2`를 개별 소스 노드로 분리하여 절삭 없이 모든 연결을 가중합으로 처리.
        - INITIAL_RECURRENCE_MODE 설정에 따라 초기 순환 간선 시딩 방식을 제어:
          * "seeded_positive"(기본): 기존 동작과 동일하게 양수 순환 가중치 시딩
          * "zero_mean": 동일한 초기 간선 구조이나 가중치를 [-1.0, 1.0] 대칭 구간에서 초기화
          * "none": 0세대 초기 순환 간선을 생성하지 않음
        """
        module_ids = list(self.modules.keys())
        rec_mode = CONFIG.get("INITIAL_RECURRENCE_MODE", "seeded_positive")
        can_seed_rec = getattr(self, "allow_recurrent_edges", self.allow_recurrent) and (
            rec_mode != "none"
        )

        for i, mod_id in enumerate(module_ids):
            # 각 초기 모듈은 3개의 센서 입력과 가중 연결됨
            for s_idx in range(3):
                src = f"in_{s_idx}"
                self.connections[mod_id][src] = random.uniform(-1.5, 1.5)
                INNOVATION_TRACKER.get_id(("edge", src, mod_id, False))

            # 계층적 순방향 연결
            if i > 0:
                prev_mod = module_ids[i - 1]
                self.connections[mod_id][prev_mod] = random.uniform(-1.2, 1.2)
                INNOVATION_TRACKER.get_id(("edge", prev_mod, mod_id, False))

            # 순환 허용 시 초기 자기 순환(Self-Recurrent) 시드 부여
            if can_seed_rec and random.random() < 0.7:
                if rec_mode == "zero_mean":
                    w_self_mod = random.uniform(-1.0, 1.0)
                else:
                    w_self_mod = random.uniform(0.4, 1.1)
                self.recurrent_connections[mod_id][mod_id] = w_self_mod
                INNOVATION_TRACKER.get_id(("edge", mod_id, mod_id, True))

        # 출력 노드(out_0, out_1, out_2) 연결
        for out_idx in range(3):
            out_key = f"out_{out_idx}"
            primary_mod = module_ids[out_idx % len(module_ids)]
            aux_mod = module_ids[(out_idx + 1) % len(module_ids)]
            self.connections[out_key][primary_mod] = random.uniform(0.6, 1.5)
            self.connections[out_key][aux_mod] = random.uniform(-1.0, 1.0)
            INNOVATION_TRACKER.get_id(("edge", primary_mod, out_key, False))
            INNOVATION_TRACKER.get_id(("edge", aux_mod, out_key, False))

            if can_seed_rec:
                # 이전 시점의 출력/모듈 상태로부터의 순환 피드백 연결
                if rec_mode == "zero_mean":
                    w_mod_out = random.uniform(-1.0, 1.0)
                    w_out_out = random.uniform(-1.0, 1.0)
                else:
                    w_mod_out = random.uniform(0.4, 1.2)
                    w_out_out = random.uniform(0.5, 1.2)
                self.recurrent_connections[out_key][primary_mod] = w_mod_out
                self.recurrent_connections[out_key][out_key] = w_out_out
                INNOVATION_TRACKER.get_id(("edge", primary_mod, out_key, True))
                INNOVATION_TRACKER.get_id(("edge", out_key, out_key, True))

    def reset_memory(self):
        """에피소드 시작 시 작업 기억(t-1 상태) 및 활성 흔적 초기화"""
        self.prev_outputs = {k: 0.0 for k in self.modules}
        for i in range(3):
            self.prev_outputs[f"in_{i}"] = 0.0
            self.prev_outputs[f"out_{i}"] = 0.0
        self.last_traces = {}

    def forward(self, stimuli):
        """
        단일 타임스텝 순방향 + 시간 지연 순환(t-1 -> t) 연산:
        - 순방향 간선(self.connections)은 위상 정렬 순서대로 현재 시점(t) 값을 전달.
        - 순환 간선(self.recurrent_connections)은 이전 시점(t-1) 출력(`self.prev_outputs`)을 전달하여
          무한 루프 없이 임의의 순환 피드백 회로를 정확히 계산.
        """
        use_rec_edges = getattr(self, "allow_recurrent_edges", self.allow_recurrent)
        use_mod_mem = getattr(self, "allow_module_memory", self.allow_recurrent)

        order = self._topological_sort()
        curr_outputs = {
            "in_0": stimuli[0],
            "in_1": stimuli[1],
            "in_2": stimuli[2],
        }
        step_traces = {}

        for node in order:
            if node not in self.modules:
                continue

            weighted_sum = 0.0
            node_trace = {}

            # 1. 현재 시점(t) 순방향 입력 합산 (절삭 없이 모든 연결 반영)
            for src, weight in self.connections.get(node, {}).items():
                src_val = curr_outputs.get(src, 0.0)
                weighted_sum += src_val * weight
                node_trace[("ff", src)] = src_val

            # 2. 이전 시점(t-1) 순환 기억 입력 합산
            if use_rec_edges:
                for src, weight in self.recurrent_connections.get(node, {}).items():
                    prev_val = self.prev_outputs.get(src, 0.0)
                    weighted_sum += prev_val * weight
                    node_trace[("rec", src)] = prev_val

            prev_self = self.prev_outputs.get(node, 0.0) if use_mod_mem else 0.0
            out_val = self.modules[node].compute(
                weighted_sum,
                prev_self_state=prev_self,
                allow_memory=use_mod_mem,
            )
            curr_outputs[node] = out_val
            step_traces[node] = (node_trace, out_val)

        # 3. 출력 노드(out_0, out_1, out_2) 계산
        logits = []
        for i in range(3):
            out_key = f"out_{i}"
            total = 0.0
            out_trace = {}

            for src, weight in self.connections.get(out_key, {}).items():
                src_val = curr_outputs.get(src, 0.0)
                total += src_val * weight
                out_trace[("ff", src)] = src_val

            if use_rec_edges:
                for src, weight in self.recurrent_connections.get(out_key, {}).items():
                    prev_val = self.prev_outputs.get(src, 0.0)
                    total += prev_val * weight
                    out_trace[("rec", src)] = prev_val

            curr_outputs[out_key] = total
            logits.append(total)
            step_traces[out_key] = (out_trace, total)

        # 다음 타임스텝(t+1)을 위해 현재 출력을 작업 기억 버퍼(prev_outputs)에 저장
        # (그래프 순환이 꺼져 있더라도 모듈 내부 기억이 켜져 있으면 이전 상태를 보존해야 함)
        if use_rec_edges or use_mod_mem:
            self.prev_outputs = curr_outputs.copy()
        self.last_traces = step_traces

        return logits, curr_outputs

    def _topological_sort(self):
        """순방향 연결(self.connections)에 대한 위상 정렬"""
        in_degree = defaultdict(int)
        all_nodes = set(self.modules.keys())

        for node in all_nodes:
            for src in self.connections.get(node, {}):
                if src in all_nodes:
                    in_degree[node] += 1

        queue = deque([n for n in sorted(all_nodes) if in_degree[n] == 0])
        result = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for target in sorted(all_nodes):
                if node in self.connections.get(target, {}):
                    in_degree[target] -= 1
                    if in_degree[target] == 0:
                        queue.append(target)

        if len(result) < len(all_nodes):
            result.extend([n for n in sorted(all_nodes) if n not in result])

        return result

    def _creates_cycle(self, source, target):
        """순방향 그래프(t -> t) 내 순환 여부 검사 (순환 간선은 t-1 -> t이므로 별도 처리)"""
        if source.startswith("in_") or target.startswith("out_"):
            return False
        if source == target:
            return True

        visited = set()
        queue = deque([source])
        while queue:
            node = queue.popleft()
            if node == target:
                return True
            if node in visited:
                continue
            visited.add(node)
            for prev_node in self.connections.get(node, {}):
                if prev_node in self.modules and prev_node not in visited:
                    queue.append(prev_node)
        return False

    def count_recurrent_edges(self):
        return sum(len(v) for v in self.recurrent_connections.values())
