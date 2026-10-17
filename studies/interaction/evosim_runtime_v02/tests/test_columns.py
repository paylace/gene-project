"""관측 타입·오류·확장 필드·손상·원자 Store 호환을 검사한다. 정책 호출0."""
import copy
import tempfile
import unittest
from pathlib import Path
from evosim_runtime_v02.storage import TableCodec, Store, canonical, digest, pack_column, unpack_column


class ColumnTests(unittest.TestCase):
    def sample(self):
        return {'genome': {'unknown': [1.]}, 'records': [
            dict(episode_id='bad-original-id', repeat=0, actions='9', rewards_cents=[12345, -1], success=False),
            dict(episode_id='0:1,2,3', repeat=1, actions='121212', rewards_cents=[99] * 6, success=True)]}

    def test_original_errors_and_unknown_fields_preserved(self):
        value = self.sample()
        value['records'][0]['future'] = {'x': [None, 1., True]}
        codec = TableCodec()
        self.assertEqual(canonical(value), canonical(codec.decode(codec.encode(value))))

    def test_exact_json_type_and_integer_boundaries(self):
        for values in ([True, False], [-128, 127], [-32768, 32767], [-32769, 32768],
                       [True, 1], [1., 0.], [[], []], [[1, 2], []], [None, 'a'], [2**70]):
            self.assertEqual(canonical(values), canonical(unpack_column(pack_column(values))))

    def test_missing_or_non_environment_records_preserved(self):
        codec = TableCodec()
        for value in ({'records': []}, {'records': 'opaque'}, {'records': [{'only_future': 1}]}):
            self.assertEqual(canonical(value), canonical(codec.decode(codec.encode(value))))

    def test_tamper_and_resigned_original_change_rejected(self):
        codec = TableCodec()
        bad = codec.encode(self.sample())
        bad['tables']['actions'][0] = '0'
        with self.assertRaises(ValueError):
            codec.decode(bad)
        bad['content_sha256'] = digest({k: v for k, v in bad.items() if k != 'content_sha256'})
        with self.assertRaises(ValueError):
            codec.decode(bad)

    def test_malformed_binary_columns_rejected(self):
        for column in ({'dtype': '|i1', 'count': 2, 'data': 'AA=='},
                       {'dtype': '<f8', 'count': 0, 'data': ''},
                       {'dtype': '|i1', 'count': 1, 'data': 'AA==', 'lengths': [2]}):
            with self.assertRaises(ValueError):
                unpack_column(column)

    def test_store_roundtrip_uses_shared_atomic_layer(self):
        parent = Path(__file__).resolve().parents[4] / 'tools/.verification_tmp'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to(parent.resolve()))
            store = Store(TableCodec(), level=9)
            path = Path(directory) / 'observed.json.gz'
            store.write(path, self.sample())
            self.assertEqual(canonical(store.read(path)), canonical(self.sample()))
            with self.assertRaises(ValueError):
                store.write(path, {})


if __name__ == '__main__':
    unittest.main()
