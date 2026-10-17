"""환경 codec을 주입하는 손실 없는 테이블 저장과 원자 파일 교체."""
import copy
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path

RECORD_KEYS = frozenset(('records', 'training_records', 'validation_records'))


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


class TableCodec:
    schema = 'lossless-tables-1'

    def __init__(self, records=None):
        self.records = records

    def encode(self, value):
        if self.records is None:
            body = {'schema': self.schema, 'record_schema': None, 'payload': value}
        else:
            tables = {'genomes': [], 'orders': [], 'actions': []}
            indexes = {key: {} for key in tables}

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
                        result[key] = {'genome_ref': intern('genomes', data)}
                    elif key in RECORD_KEYS:
                        if not data:
                            result[key] = {'literal': data}
                            continue
                        if any(not isinstance(r, dict) or not {'episode_id', 'repeat', 'actions'} <= set(r) for r in data):
                            raise ValueError('필수 증거 필드 누락')
                        order = [[r['episode_id'], r['repeat']] for r in data]
                        if any(type(rep) is not int or rep < 0 for _, rep in order):
                            raise ValueError('잘못된 반복 번호')
                        actions = [r['actions'] for r in data]
                        training = all('progress_units' in r for r in data)
                        try:
                            reconstructed = self.records.restore_unique(order, actions, training)
                        except (ValueError, TypeError, RuntimeError, AttributeError):
                            # 잘못된 실제 관측 자체가 증거다. 조용히 고치거나 버리지 않는다.
                            result[key] = {'literal': data}
                            continue
                        residuals = []
                        for row, expected in zip(data, reconstructed):
                            expected.update(episode_id=row['episode_id'], repeat=row['repeat'])
                            changes = {k: v for k, v in row.items() if k not in expected or canonical(v) != canonical(expected[k])}
                            missing = [k for k in expected if k not in row]
                            residuals.append([changes, missing])
                        result[key] = {'order': intern('orders', order), 'training': training,
                            'action_refs': [intern('actions', a) for a in actions]}
                        if any(change or missing for change, missing in residuals):
                            result[key]['residuals'] = residuals
                    else:
                        result[key] = visit(data)
                return result

            payload = visit(value)
            body = {'schema': self.schema, 'record_schema': self.records.schema,
                    'dependencies': self.records.dependencies, 'tables': tables, 'payload': payload}
        body['original_sha256'] = digest(value)
        return {**body, 'content_sha256': digest(body)}

    def decode(self, envelope):
        body = {k: v for k, v in envelope.items() if k != 'content_sha256'}
        if digest(body) != envelope['content_sha256'] or body['schema'] != self.schema:
            raise ValueError('저장 내용/schema 불일치')
        if body['record_schema'] is None:
            result = body['payload']
        else:
            if (self.records is None or body['record_schema'] != self.records.schema
                    or body['dependencies'] != self.records.dependencies):
                raise ValueError('복원 환경/schema 변경')
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
                        result[key] = copy.deepcopy(ref('genomes', data['genome_ref']))
                    elif key in RECORD_KEYS:
                        if 'literal' in data:
                            result[key] = copy.deepcopy(data['literal'])
                            continue
                        order = ref('orders', data['order'])
                        actions = [ref('actions', i) for i in data['action_refs']]
                        if len(order) != len(actions):
                            raise ValueError('행동 목록 길이 불일치')
                        records = self.records.restore_unique(order, actions, data['training'])
                        residuals = data.get('residuals', [[{}, []] for _ in order])
                        if len(residuals) != len(order):
                            raise ValueError('예외 목록 길이 불일치')
                        for row, (name, rep), (changes, missing) in zip(records, order, residuals):
                            row.update(episode_id=name, repeat=rep)
                            row.update(copy.deepcopy(changes))
                            for field in missing:
                                del row[field]
                        result[key] = records
                    else:
                        result[key] = visit(data)
                return result
            result = visit(body['payload'])
        if digest(result) != body['original_sha256']:
            raise ValueError('원본 전체 내용 해시 불일치')
        return result


class Store:
    def __init__(self, codec=None, level=3):
        if type(level) is not int or not 0 <= level <= 9:
            raise ValueError('잘못된 gzip 강도')
        self.codec, self.level = codec, level

    def encode(self, value):
        return canonical(self.codec.encode(value) if self.codec else value)

    def read(self, path):
        path = Path(path)
        raw = path.read_bytes()
        value = json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)
        return self.codec.decode(value) if self.codec else value

    def write(self, path, value, *, replace=False):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not replace:
            if canonical(self.read(path)) != canonical(value):
                raise ValueError('완료 기록 덮어쓰기 금지')
            return False
        data = self.encode(value)
        if path.suffix == '.gz':
            data = gzip.compress(data, compresslevel=self.level, mtime=0)
        fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.', suffix='.tmp')
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            if replace:
                os.replace(name, path)
            else:
                # 독점 생성: 다른 작업자가 같은 완료 파일을 덮어쓰지 못한다.
                if os.name == 'nt':
                    # Windows rename은 기존 대상이 있으면 실패한다. 하드링크 권한 불필요.
                    os.rename(name, path)
                else:
                    os.link(name, path)
        finally:
            Path(name).unlink(missing_ok=True)
        return True
