"""한 회차 번식만 구현. 탐색 실행·검증 후보 선발·재개는 W4에서 추가한다."""

from .network import crossover, finite_number, mutate


def next_population(population, scores, rng, allocator, settings):
    if not population or len(scores) != len(population) or not all(finite_number(value) for value in scores):
        raise ValueError("Invalid population/scores")
    for genome in population:
        genome.validate(settings)
    allocator.validate(population)
    p = settings.parameters
    ranked = sorted(range(len(population)), key=lambda idx: (-scores[idx], idx))
    children, ancestry = [], []
    elites = max(1, int(len(population) * p["elite_ratio"]))
    for idx in ranked[:elites]:
        children.append(population[idx].clone())
        ancestry.append({"kind": "elite", "parents": [idx], "mutation": None,
                         "parent_digests": [population[idx].digest], "dominant_parent_slot": idx,
                         "slot": len(children) - 1, "child_digest": children[-1].digest,
                         "condition": settings.condition, "settings_digest": settings.digest, "mutation_applied": False})

    def select():
        slots = rng.sample(range(len(population)), min(p["tournament_size"], len(population)))
        return max(slots, key=lambda idx: (scores[idx], -idx))

    while len(children) < len(population):
        first = select()
        parents = [first]
        dominant = first
        if rng.random() < p["crossover_rate"]:
            second = select()
            parents.append(second)
            dominant = max(parents, key=lambda idx: (scores[idx], -idx))
            child = crossover(population[first], population[second], rng, scores[first], scores[second], settings, first, second)
            kind = "crossover"
        else:
            child = population[first].clone()
            kind = "clone"
        mutation = mutate(child, rng, allocator, settings) if rng.random() < p["mutation_rate"] else None
        children.append(child)
        ancestry.append({"kind": kind, "parents": parents, "mutation": mutation,
                         "parent_digests": [population[idx].digest for idx in parents], "dominant_parent_slot": dominant,
                         "slot": len(children) - 1, "child_digest": child.digest,
                         "condition": settings.condition, "settings_digest": settings.digest,
                         "mutation_applied": mutation is not None})
    return children, ancestry
