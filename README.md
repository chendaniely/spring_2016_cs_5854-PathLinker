# PathLinker

Code to develop PathLinker 2.0

The Computational Systems Biology course is taught by [T. M. Murali](http://bioinformatics.cs.vt.edu/~murali/)
at Virginia Tech

Original PathLiker repository: https://github.com/Murali-group/PathLinker

## Getting `master-script.py` working
`master-script.py` is in the murali group server.
So it's not publically availiable.
This script is just wrapers setting up data and options to run various algorithims, including `pathlinker`.

- add `svnrepo/src/python` to the `PYTHONPATH`.
```
PYTHONPATH="/home/chend/svn/svnrepo/src/python/scripts/trunk:$PYTHONPATH"
export PYTHONPATH
```
- add a softlink to `data -> /data/annaritz/projects/2015-03-pathlinker/data/` in the root `PathLinker` directory.
- example `master-script.py` call: `python master-script.py --pathlinker -- netpath -- k 10 --weightedppi`
  - `python master-script.py --help` helps a lot
