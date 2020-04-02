from pathlib import Path

p = Path('/data/work/cell').glob("*.png")
for item in p:
    a = str(item).split('/')[-1].split('.')[0].replace('$','/')
    print(item)
    print(a)

p = Path('/data/work/cell').glob("*.png")
for item in p:
    label=str(item).split('/')[-1].split('.')[0]