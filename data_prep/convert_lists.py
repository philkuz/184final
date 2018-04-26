from glob import glob
import os
pj = os.path.join
lists_dir = 'lists/geometry-v1'
out_dir = '../data/geometry-v1'
files = glob(pj(lists_dir, '*.txt'))
for fpath in files:
  out_path = pj(out_dir, fpath.split('/')[-1])
  with open(fpath) as f, open(out_path, 'w') as out_f:
    for img_path in f:
      paths = img_path.strip().split(',')
      out_paths = []
      for p in paths:
        path_split = p.split('/')
        base_path ='/'.join(path_split[path_split.index('data'):])
        print(base_path)
        out_paths.append(pj('../', base_path))
      out_f.write('{}\n'.format(','.join(out_paths)))
