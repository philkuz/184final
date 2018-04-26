class yield_list:
  """ Decorator that turns a generator into a function that returns a list """
  def __init__(self, f):
    self.f = f
  def __call__(self, *args, **kwargs):
    return [x for x in self.f(*args, **kwargs)]
@yield_list
def split_n(xs, n):
  n = int(n)
  assert n > 0
  i = 0
  while i < len(xs):
    yield xs[i : i + n]
    i += n

def parmap(f, xs, nproc = None):
  import pathos.multiprocessing
  if nproc is None:
    nproc = 12
  pool = pathos.multiprocessing.Pool(processes = nproc)
  try:
    ret = pool.map_async(f, xs).get(10000000)
  finally:
    pool.close()
  return ret
