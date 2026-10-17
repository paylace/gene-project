"""공통 tanh 순방향 표현과 조건별 탐색 연산. 과거 소스는 변경하지 않는다."""
import random
from evosim_interaction_v02.network import (
    Genome as BaseGenome, Module, ModuleIds, INPUTS, OUTPUTS, KINDS,
    node_key, add_ff_if_acyclic, crossover as original_crossover)
from evosim_interaction_v02.settings import Settings

NETWORK = Settings('A2')


class Genome(BaseGenome):
    def validate(self, settings=None):
        super().validate(settings)
        if self.settings != NETWORK or self.rec or not 4 <= len(self.modules) <= 9:
            raise ValueError('Q1-006 requires four to nine modules and no recurrence')
        if any(m.op != 'tanh' or m.memory_decay != .5 or not -2 <= m.bias <= 2 for m in self.modules.values()):
            raise ValueError('Only tanh with fixed inert decay and bounded bias is allowed')
        if any(not -3 <= w <= 3 for edges in self.ff.values() for w in edges.values()):
            raise ValueError('Weight outside registered bounds')


def topology(genome):
    return (tuple(sorted(genome.modules, key=node_key)),
            tuple((t, tuple(sorted(edges, key=node_key))) for t, edges in sorted(genome.ff.items(), key=lambda x: node_key(x[0]))))


def validate_condition(genome, condition):
    genome.validate(NETWORK)
    if condition not in ('F0', 'F1'):
        raise ValueError('Unknown search condition')
    if condition == 'F0':
        expected = {'m'+str(i): set(INPUTS) for i in range(4)}
        expected.update({o: {'m'+str(i) for i in range(4)} for o in OUTPUTS})
        if set(genome.modules) != {'m'+str(i) for i in range(4)} or {t:set(e) for t,e in genome.ff.items()} != expected:
            raise ValueError('F0 topology changed')


def initial_genome(rng, settings=NETWORK):
    if settings != NETWORK:
        raise ValueError('Unexpected network setting')
    modules={f'm{i}':Module('tanh',rng.uniform(-.4,.4),.5) for i in range(4)}
    ff={m:{src:rng.uniform(-1.5,1.5) for src in INPUTS} for m in modules}
    ff.update({out:{m:rng.uniform(-1,1) for m in modules} for out in OUTPUTS})
    genome=Genome(modules,ff,{},NETWORK)
    genome.validate()
    return genome


def mutate(genome, rng, allocator, condition, forced_kind=None):
    validate_condition(genome, condition)
    allocator.validate((genome,))
    kind=forced_kind or rng.choices(KINDS, NETWORK.parameters['mutation_weights'], k=1)[0]
    if kind not in KINDS:
        raise ValueError('Unknown mutation')
    before=genome.digest
    rejected=[]
    added=None
    ids=sorted(genome.modules,key=node_key)
    if kind=='weight':
        for node in rng.sample(ids,min(2,len(ids))):
            m=genome.modules[node]
            bias=m.bias
            if rng.random()<.35:
                bias=max(-2,min(2,bias+rng.gauss(0,.25)))
            genome.modules[node]=Module('tanh',bias,.5)
        for target in sorted(genome.ff,key=node_key):
            for src in sorted(genome.ff[target],key=node_key):
                if rng.random()<.35:
                    genome.ff[target][src]=max(-3,min(3,genome.ff[target][src]+rng.gauss(0,.35)))
    elif condition=='F0' or kind=='add_rec':
        rejected.append({'reason':'condition_forbidden','proposal':kind})
    elif kind=='add_module' and len(ids)<9:
        added=allocator.allocate()
        genome.modules[added]=Module('tanh',rng.uniform(-.4,.4),.5)
        source=rng.choice(ids+list(INPUTS))
        genome.ff[added]={source:rng.uniform(-1.2,1.2)}
        genome.ff.setdefault(rng.choice(OUTPUTS),{})[added]=rng.uniform(-1.2,1.2)
    elif kind=='remove_module' and len(ids)>4:
        removed=rng.choice(ids)
        del genome.modules[removed]
        genome.ff.pop(removed,None)
        for edges in genome.ff.values():
            edges.pop(removed,None)
        for out in OUTPUTS:
            if not genome.ff.get(out):
                genome.ff[out]={rng.choice(sorted(genome.modules,key=node_key)):rng.uniform(-1,1)}
    elif kind=='add_ff':
        target=rng.choice(ids+list(OUTPUTS))
        source=rng.choice(ids+list(INPUTS))
        if source in genome.ff.get(target,{}):
            rejected.append({'reason':'duplicate_edge','proposal':kind})
        elif not add_ff_if_acyclic(genome,source,target,rng.uniform(-1.2,1.2)):
            rejected.append({'reason':'ff_cycle','proposal':kind})
    elif kind=='remove_edge':
        # 원 제안처럼 recurrent mapping을 고른 경우 없는 간선으로 끝난다.
        if rng.random()<.4:
            rejected.append({'reason':'no_removable_target','proposal':kind})
        else:
            targets=sorted((t for t,e in genome.ff.items() if len(e)>1),key=node_key)
            if targets:
                target=rng.choice(targets)
                del genome.ff[target][rng.choice(sorted(genome.ff[target],key=node_key))]
            else:
                rejected.append({'reason':'no_removable_target','proposal':kind})
    else:
        rejected.append({'reason':'module_limit','proposal':kind})
    validate_condition(genome,condition)
    if before==genome.digest and not rejected:
        rejected.append({'reason':'no_parameter_change','proposal':kind})
    return {'kind':kind,'added_module':added,'before':before,'after':genome.digest,
            'changed':before!=genome.digest,'proposal_rejected':any(r['reason']=='condition_forbidden' for r in rejected),
            'rejections':rejected}


def crossover(first,second,rng,fitness1,fitness2,condition,slot1=0,slot2=1):
    validate_condition(first,condition)
    validate_condition(second,condition)
    child=original_crossover(first,second,rng,fitness1,fitness2,NETWORK,slot1,slot2)
    validate_condition(child,condition)
    return child


def next_population(population,scores,rng,allocator,condition):
    if not population or len(population)!=len(scores) or any(not isinstance(s,(int,float)) for s in scores):
        raise ValueError('Invalid population fitness')
    for g in population:validate_condition(g,condition)
    allocator.validate(population)
    ranked=sorted(range(len(scores)),key=lambda i:(-scores[i],i))
    children=[];ancestry=[]
    def append(child,kind,parents,dominant,mutation):
        validate_condition(child,condition)
        children.append(child)
        ancestry.append({'kind':kind,'parents':parents,'mutation':mutation,
          'parent_digests':[population[i].digest for i in parents],'dominant_parent_slot':dominant,
          'slot':len(children)-1,'child_digest':child.digest,'condition':condition,
          'settings_digest':NETWORK.digest,'mutation_applied':mutation is not None})
    for i in ranked[:max(1,int(len(population)*.2))]:
        append(population[i].clone(),'elite',[i],i,None)
    def select():
        return max(rng.sample(range(len(population)),min(3,len(population))),key=lambda i:(scores[i],-i))
    while len(children)<len(population):
        a=select();parents=[a];dominant=a
        if rng.random()<.65:
            b=select();parents.append(b);dominant=max(parents,key=lambda i:(scores[i],-i))
            child=crossover(population[a],population[b],rng,scores[a],scores[b],condition,a,b);kind='crossover'
        else:
            child=population[a].clone();kind='clone'
        mutation=mutate(child,rng,allocator,condition) if rng.random()<.5 else None
        append(child,kind,parents,dominant,mutation)
    return children,ancestry
