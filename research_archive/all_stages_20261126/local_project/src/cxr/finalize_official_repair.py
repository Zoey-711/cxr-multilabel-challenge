"""Finalize the independently verified build kit and validate the ZIP bytes."""
import argparse
import ast
import json
import shutil
import zipfile
from pathlib import Path

from cxr.official_deployment_tools import lock_manifest, sha256, verify_package, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    root=args.config.resolve().parent
    project=root.parents[2]
    local_env=root.parent/'official_repair_local_environment_20261110.json'
    shutil.copy2(local_env,root/'materials/local_environment_blocked.json')
    shutil.copy2(project/'tests/test_official_deployment_tools.py',root/'materials/test_official_deployment_tools.py')
    shutil.copy2(project/'tests/test_official_repaired_cost.py',root/'materials/test_official_repaired_cost.py')
    # Preserve verifiable upstream compatibility reference; supplied organizer image remains authoritative.
    (root/'materials/compatibility_sources.md').write_text(
        '# 环境来源与兼容性\n\n'
        '官方镜像及版本来自用户2026-11-10提供的服务器参数，见official_environment.json。\n'
        'TensorRT引擎不能假定可由较旧运行库加载，因此旧11.4引擎不进入修订包。\n'
        'NVIDIA官方说明：https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html\n'
        '本包没有把最新版文档当作TensorRT10.9实测证明；10.9构建尚待官方镜像执行。\n',encoding='utf-8')
    (root/'final_decision.md').write_text(
        '# 官方环境修订完成；目标验收未完成\n\n'
        '队名：王卓颖。历史78.81提交CSV、六个模型的训练权重、384输入、类别顺序、'
        'Study Mean及固定融合均保持冻结。预测与预处理函数AST比对通过。\n\n'
        '20项CPU测试通过，其中真实cost循环以CPU模拟执行305次推理调用（5预热+300计时）；'
        '模拟不表示GPU性能。两个FP32 ONNX均通过结构及冻结checkpoint的固定合成输入对照，'
        '最大概率差分别为B02 4.76837158203125e-7、M02 4.172325134277344e-7。\n\n'
        'Windows环境门禁正确阻止正式构建。未安装Docker/TensorRT，未执行GPU推理或训练。'
        '旧环境TensorRT11.4引擎未沿用；新引擎须在官方镜像TensorRT10.9 T4上生成，'
        '源码提供构建后合成对照及全Test审计，不能保证编译优化保持历史预测排序。\n\n'
        '本ZIP是构建交接包，含原checkpoint和ONNX构建源，解压后可超过500MB；'
        'build_engines.py输出的独立candidate目录自动排除构建源，并检查其完整文件总大小。'
        '当前不能引用旧引擎的392800139字节作为新引擎的大小结论。\n\n'
        '正式递交仍待：官方环境实际加载、新引擎预测一致性、官方100图测速、'
        '部分Study及Gate口径确认、预训练许可材料、更新版声明模板核对与签署。\n\n'
        '结论：PREPARED_PENDING_T4_BUILD / NOT_READY_FOR_OFFICIAL_ACCEPTANCE。\n',encoding='utf-8')
    py_count=0
    for path in root.rglob('*.py'):
        ast.parse(path.read_text(encoding='utf-8-sig'))
        py_count+=1
    report=json.loads((root/'materials/local_verification.json').read_text(encoding='utf-8'))
    report['python_syntax_files']=py_count
    report['official_environment_negative_test']='PASS: blocked Windows/4060/missing TensorRT'
    write_json(root/'materials/local_verification.json',report)
    records=lock_manifest(root)
    verify_package(root)
    archive=root.with_suffix('.zip')
    if archive.exists():
        raise FileExistsError(archive)
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(root.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                if path.suffix.lower() in ('.jpg','.jpeg','.png','.dcm','.engine'):
                    raise ValueError(f'Unexpected raw image/legacy engine: {path}')
                z.write(path,'official_kit/'+path.relative_to(root).as_posix())
    import hashlib
    with zipfile.ZipFile(archive) as z:
        members=z.namelist()
        if len(members)!=len(set(members)) or any('..' in Path(n).parts for n in members):
            raise ValueError('Unsafe archive')
        for relative,expected in records.items():
            with z.open('official_kit/'+relative) as stream:
                actual=hashlib.file_digest(stream,'sha256').hexdigest()
            if actual!=expected:
                raise ValueError('Archive content mismatch: '+relative)
        if z.testzip() is not None:
            raise ValueError('Archive CRC failed')
    result={'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha256(archive),
            'entries':len(members),'archive_content_hashes':'PASS','package_file_hashes':len(records),
            'submission_sha256':sha256(root/'submission.csv'),
            'uncompressed_build_kit_bytes':sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts),
            'status':'PREPARED_PENDING_T4_BUILD','t4_status':'UNVERIFIED'}
    write_json(archive.with_suffix('.zip.sha256.json'),result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
