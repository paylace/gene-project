"""완료한 Q1-009 원자료와 동결 소스를 별도 ZIP으로 보존한다."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"studies/interaction"))
from evosim_q1009_v01.protocol import load,RESULTS
from evosim_interaction_v02.storage import read_json,file_digest


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--catalog',type=Path,required=True)
    args=parser.parse_args();output,catalog=args.output.resolve(),args.catalog.resolve()
    if not output.is_relative_to(ROOT) or not catalog.is_relative_to(ROOT) or output.exists() or catalog.exists():
        raise ValueError('Use new workspace output/catalog paths')
    config,_=load(RESULTS/'main');summary=read_json(RESULTS/'main/summary.json')
    if summary['sample_size']!=500 or summary['source_digest']!=config['source_digest'] or len(summary['artifacts'])!=505:
        raise ValueError('Incomplete official cohort')
    for name,hashes in summary['artifacts'].items():
        for kind,path in (('raw',RESULTS/'main/raw'/(name+'.json.gz')),('audit',RESULTS/'main/audits'/(name+'.json'))):
            if file_digest(path)!=hashes[kind]:raise ValueError('Completed artifact changed')
    source=read_json(ROOT/'docs/management/q1_009_source_archive.json')
    if sha(ROOT/source['archive'])!=source['archive_sha256']:raise ValueError('Frozen source archive changed')
    files=set(p for p in RESULTS.rglob('*') if p.is_file())
    files.add(ROOT/source['archive'])
    files.update((ROOT/'docs/management').glob('q1_009_*.json'))
    files.update((ROOT/'docs/management').glob('q1_009_*.csv'))
    files.update((ROOT/'docs/management').glob('q1_009_*.log'))
    files.update((ROOT/'dist').glob('q1009_*.log'))
    files.update(p for p in (ROOT/'dist/q1009_gate').rglob('*') if p.is_file())
    files.add(ROOT/'docs/protocols/q1_009_v01_plan.json')
    partial=output.with_suffix(output.suffix+'.partial')
    manifest={}
    with zipfile.ZipFile(partial,'x',allowZip64=True) as archive:
        for index,path in enumerate(sorted(files)):
            name=path.relative_to(ROOT).as_posix();before=path.stat();digest=hashlib.sha256();size=0
            entry=zipfile.ZipInfo(name);entry.compress_type=zipfile.ZIP_STORED if path.suffix in ('.gz','.zip') else zipfile.ZIP_DEFLATED
            with path.open('rb') as source,archive.open(entry,'w',force_zip64=True) as target:
                for block in iter(lambda:source.read(1024*1024),b''):
                    digest.update(block);size+=len(block);target.write(block)
            if (before.st_size,before.st_mtime_ns)!=(path.stat().st_size,path.stat().st_mtime_ns):raise ValueError('Input changed during packaging')
            manifest[name]={'bytes':size,'sha256':digest.hexdigest()}
            if (index+1)%500==0:print('ARCHIVE',index+1,'/',len(files),flush=True)
        payload=json.dumps({'files':manifest},ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')
        archive.writestr('MANIFEST.json',payload,compress_type=zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(partial) as archive:
        if archive.testzip():raise ValueError('Archive CRC mismatch')
    partial.rename(output)
    result={'experiment_id':'Q1-009','archive':output.relative_to(ROOT).as_posix(),'bytes':output.stat().st_size,
            'sha256':sha(output),'files':len(manifest),'uncompressed_bytes':sum(v['bytes'] for v in manifest.values()),
            'manifest_sha256':hashlib.sha256(payload).hexdigest(),'source_digest':config['source_digest'],
            'protocol_digest':config['protocol_digest'],'publication':'local archive; publication pending',
            'integrity_check':'all input SHA-256 records and ZIP CRC passed'}
    catalog.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
