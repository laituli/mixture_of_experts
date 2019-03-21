import time
import sys

for i in range(10):
    print('\r%d' % i, end='')
    time.sleep(0.5)
    sys.stdout.flush()