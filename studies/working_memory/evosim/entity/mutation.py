"""Entity의 구조적 변이 담당: 가중치/모듈/순방향 간선/순환 간선 돌연변이"""

import random

from ..config import CONFIG
from ..innovation import INNOVATION_TRACKER
from ..module import Module


class MutationMixin:
    def mutate(self):
        """가중치, 모듈, 순방향 간선 및 순환(Recurrent) 간선 돌연변이"""
        use_rec_edges = getattr(self, "allow_recurrent_edges", self.allow_recurrent)
        mutation_choices = [
            "weight",
            "add_module",
            "remove_module",
            "add_ff_connection",
            "add_recurrent_connection",
            "remove_connection",
        ]
        rec_prob = CONFIG["RECURRENT_MUTATION_PROB"] if use_rec_edges else 0.0
        weights = [0.42, 0.12, 0.06, 0.18, rec_prob, 0.10]

        mutation_type = random.choices(mutation_choices, weights, k=1)[0]

        if mutation_type == "weight":
            if self.modules:
                for mod in random.sample(
                    list(self.modules.values()), min(2, len(self.modules))
                ):
                    mod.mutate()

            for conn_dict in (self.connections, self.recurrent_connections):
                for target, src_map in conn_dict.items():
                    for src in list(src_map.keys()):
                        if random.random() < 0.35:
                            src_map[src] = max(
                                -3.0, min(3.0, src_map[src] + random.gauss(0, 0.35))
                            )

        elif mutation_type == "add_module" and len(self.modules) < CONFIG["MAX_MODULES"]:
            existing_modules = list(self.modules.keys())
            next_idx = max([int(k[1:]) for k in existing_modules], default=-1) + 1
            new_id = f"m{next_idx}"
            self.modules[new_id] = Module()
            INNOVATION_TRACKER.get_id(("module", new_id))

            src_candidates = existing_modules + ["in_0", "in_1", "in_2"]
            chosen_src = random.choice(src_candidates)
            self.connections[new_id][chosen_src] = random.uniform(-1.2, 1.2)
            INNOVATION_TRACKER.get_id(("edge", chosen_src, new_id, False))

            if use_rec_edges and random.random() < 0.5:
                self.recurrent_connections[new_id][new_id] = random.uniform(0.4, 1.0)
                INNOVATION_TRACKER.get_id(("edge", new_id, new_id, True))

            out_target = f"out_{random.randint(0, 2)}"
            self.connections[out_target][new_id] = random.uniform(-1.2, 1.2)
            INNOVATION_TRACKER.get_id(("edge", new_id, out_target, False))

        elif mutation_type == "remove_module" and len(self.modules) > CONFIG["INITIAL_MODULES"]:
            to_remove = random.choice(list(self.modules.keys()))
            del self.modules[to_remove]
            self.connections.pop(to_remove, None)
            self.recurrent_connections.pop(to_remove, None)

            for conn_dict in (self.connections, self.recurrent_connections):
                for node in list(conn_dict.keys()):
                    conn_dict[node].pop(to_remove, None)

            for i in range(3):
                out_key = f"out_{i}"
                if not self.connections.get(out_key) and self.modules:
                    fallback = random.choice(list(self.modules.keys()))
                    self.connections[out_key][fallback] = random.uniform(-1.0, 1.0)

        elif mutation_type == "add_ff_connection":
            targets = list(self.modules.keys()) + ["out_0", "out_1", "out_2"]
            sources = list(self.modules.keys()) + ["in_0", "in_1", "in_2"]
            target = random.choice(targets)
            source = random.choice(sources)

            if source not in self.connections[target] and not self._creates_cycle(
                source, target
            ):
                self.connections[target][source] = random.uniform(-1.2, 1.2)
                INNOVATION_TRACKER.get_id(("edge", source, target, False))

        elif mutation_type == "add_recurrent_connection" and use_rec_edges:
            targets = list(self.modules.keys()) + ["out_0", "out_1", "out_2"]
            sources = list(self.modules.keys()) + ["out_0", "out_1", "out_2"]
            target = random.choice(targets)
            source = random.choice(sources)

            if source not in self.recurrent_connections[target]:
                self.recurrent_connections[target][source] = random.uniform(-1.2, 1.2)
                INNOVATION_TRACKER.get_id(("edge", source, target, True))

        elif mutation_type == "remove_connection":
            conn_dict = (
                self.recurrent_connections
                if (use_rec_edges and random.random() < 0.4)
                else self.connections
            )
            valid_targets = [k for k, v in conn_dict.items() if len(v) > 1]
            if valid_targets:
                target = random.choice(valid_targets)
                drop_src = random.choice(list(conn_dict[target].keys()))
                del conn_dict[target][drop_src]
