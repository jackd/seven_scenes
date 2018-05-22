#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import seven_scenes.preprocessed.bin as b

for mode in ('test', 'train'):
    noisy, clean = zip(*b.get_filename_split(mode))
    print('%s filenames (noisy, clean): %d, %d'
          % (mode, len(noisy), len(clean)))
    print('Unique: %d, %d' % (len(set(noisy)), len(set(clean))))
