"""환경 재생 없이 실제 관측을 직접 보존하는 범용 열 codec."""
import base64
import copy
import numpy as np
from evosim_runtime_v01.storage import Store, canonical, digest, RECORD_KEYS


def pack_column(values):
    """정확히 같은 타입/범위만 이진화한다. 나머지는 원래 JSON으로 보존한다."""
    if not values:
        return {'json': values}
    flat, lengths = values, None
    if all(type(v) is list for v in values):
        lengths = [len(v) for v in values]
        flat = [x for v in values for x in v]
    if not flat:
        return {'json': values}
    if all(type(v) is bool for v in flat):
        dtype = '|b1'
    elif all(type(v) is int and -128 <= v <= 127 for v in flat):
        dtype = '|i1'
    elif all(type(v) is int and -32768 <= v <= 32767 for v in flat):
        dtype = '<i2'
    else:
        return {'json': values}
    result = {'dtype': dtype, 'count': len(flat),
              'data': base64.b64encode(np.array(flat, dtype=dtype).tobytes()).decode('ascii')}
    if lengths is not None:
        result['lengths'] = lengths
    return result


def unpack_column(column):
    if 'json' in column:
        return copy.deepcopy(column['json'])
    dtype = column['dtype']
    count = column['count']
    if dtype not in ('|b1', '|i1', '<i2') or type(count) is not int or count < 0:
        raise ValueError('잘못된 열 타입/길이')
    raw = base64.b64decode(column['data'], validate=True)
    if len(raw) != np.dtype(dtype).itemsize * count:
        raise ValueError('열 바이트 길이 불일치')
    values = np.frombuffer(raw, dtype=dtype).tolist()
    if 'lengths' not in column:
        return values
    lengths = column['lengths']
    if any(type(n) is not int or n < 0 for n in lengths) or sum(lengths) != count:
        raise ValueError('리스트 길이 목록 불일치')
    result, offset = [], 0
    for n in lengths:
        result.append(values[offset:offset + n])
        offset += n
    return result


class TableCodec:
    schema = 'observed-columns-1'

    def encode(self, value):
        tables = {'genomes': [], 'orders': [], 'actions': []}
        indexes = {k: {} for k in tables}

        def intern(kind, item):
            key = canonical(item)
            if key not in indexes[kind]:
                indexes[kind][key] = len(tables[kind])
                tables[kind].append(item)
            return indexes[kind][key]

        def visit(item):
            if isinstance(item, list):
                return [visit(v) for v in item]
            if not isinstance(item, dict):
                return item
            result = {}
            for key, data in item.items():
                if key == 'genome':
                    result[key] = intern('genomes', data)
                elif key in RECORD_KEYS:
                    if (not isinstance(data, list) or not data
                            or any(not isinstance(r, dict) or not {'episode_id', 'repeat', 'actions'} <= set(r) for r in data)):
                        result[key] = {'literal': data}
                        continue
                    order = [[r['episode_id'], r['repeat']] for r in data]
                    columns, sparse = {}, {}
                    fields = sorted(set().union(*(set(r) for r in data)) - {'episode_id', 'repeat', 'actions'})
                    for field in fields:
                        present = [i for i, row in enumerate(data) if field in row]
                        column = pack_column([data[i][field] for i in present])
                        if len(present) == len(data):
                            columns[field] = column
                        else:
                            sparse[field] = [present, column]
                    result[key] = {'order': intern('orders', order), 'columns': columns,
                                   'actions': [intern('actions', r['actions']) for r in data]}
                    if sparse:
                        result[key]['sparse'] = sparse
                else:
                    result[key] = visit(data)
            return result

        payload = visit(value)
        body = dict(schema=self.schema, tables=tables, payload=payload, original_sha256=digest(value))
        return {**body, 'content_sha256': digest(body)}

    def decode(self, envelope):
        body = {k: v for k, v in envelope.items() if k != 'content_sha256'}
        if body['schema'] != self.schema or digest(body) != envelope['content_sha256']:
            raise ValueError('내용/schema 체크섬 불일치')
        tables = body['tables']

        def ref(kind, index):
            if type(index) is not int or not 0 <= index < len(tables[kind]):
                raise ValueError('잘못된 참조')
            return tables[kind][index]

        def visit(item):
            if isinstance(item, list):
                return [visit(v) for v in item]
            if not isinstance(item, dict):
                return item
            result = {}
            for key, data in item.items():
                if key == 'genome':
                    result[key] = copy.deepcopy(ref('genomes', data))
                elif key in RECORD_KEYS:
                    if 'literal' in data:
                        result[key] = copy.deepcopy(data['literal'])
                        continue
                    order = ref('orders', data['order'])
                    actions = [ref('actions', i) for i in data['actions']]
                    if len(actions) != len(order):
                        raise ValueError('행동·순서 길이 불일치')
                    rows = [dict(episode_id=copy.deepcopy(ep), repeat=copy.deepcopy(rep), actions=copy.deepcopy(action))
                            for (ep, rep), action in zip(order, actions)]
                    for field, column in data['columns'].items():
                        values = unpack_column(column)
                        if len(values) != len(rows):
                            raise ValueError('열 행 수 불일치')
                        for row, value in zip(rows, values):
                            row[field] = value
                    for field, (indices, column) in data.get('sparse', {}).items():
                        values = unpack_column(column)
                        if len(values) != len(indices) or len(set(indices)) != len(indices):
                            raise ValueError('희소 열 길이/중복 불일치')
                        for index, value in zip(indices, values):
                            if type(index) is not int or not 0 <= index < len(rows):
                                raise ValueError('희소 열 참조 불일치')
                            rows[index][field] = value
                    result[key] = rows
                else:
                    result[key] = visit(data)
            return result
        result = visit(body['payload'])
        if digest(result) != body['original_sha256']:
            raise ValueError('원래 관측 내용과 다름')
        return result
