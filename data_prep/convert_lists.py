from glob import glob
import os
pj = os.path.join
# lists_dir = 'lists/geometry-v1'
lists_dir = '../data/geometry-v3'
relevant_dir = lists_dir.split('/')[-1]
contain_out_dir = '../data_scratch'
out_dir = pj(contain_out_dir, relevant_dir)
files = glob(pj(lists_dir, '*.txt'))
for fpath in files:
  out_path = pj(out_dir, fpath.split('/')[-1])
  with open(fpath) as f, open(out_path, 'w') as out_f:
    for img_path in f:
      paths = img_path.strip().split(',')
      out_paths = []
      for p in paths:
        path_split = p.split('/')
        base_path ='/'.join(path_split[path_split.index(relevant_dir):])
        out_paths.append(pj(contain_out_dir, base_path))
        print(out_paths[-1])
      out_f.write('{}\n'.format(','.join(out_paths)))
