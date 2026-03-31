# patch_basicsr.py — 在激活的 conda 环境中运行
import os, re, shutil, sys

conda_prefix = os.environ.get('CONDA_PREFIX')
if not conda_prefix:
    print("找不到 CONDA_PREFIX，请确保在 conda 环境中运行。")
    sys.exit(1)

base = os.path.join(conda_prefix, 'Lib', 'site-packages', 'basicsr')
if not os.path.isdir(base):
    print("找不到目录：", base)
    sys.exit(1)

count = 0
for root, _, files in os.walk(base):
    for f in files:
        if f.endswith('.py'):
            path = os.path.join(root, f)
            with open(path, 'r', encoding='utf-8') as fh:
                text = fh.read()
            if 'functional_tensor' in text:
                bak = path + '.bak'
                shutil.copy(path, bak)
                new = re.sub(r'torchvision\.transforms\.functional_tensor', 'torchvision.transforms.functional', text)
                with open(path, 'w', encoding='utf-8') as fh:
                    fh.write(new)
                print("Patched:", path, "-> backup:", bak)
                count += 1

print("Done. total patched files:", count)
