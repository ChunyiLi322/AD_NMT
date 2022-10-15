import os

import unicodedata
import re
import numpy as np
import os
import io
import time
import random



sentences = [] 
lines = []
with open("spa.txt", 'r') as origin_file:
	for line in origin_file:
		lines.append(line)
random.shuffle(lines)

for line in lines[0:100]:
    input_batch = line.split('\t')[0]
    output_batch = line.split('\t')[1]
    sentences.append(input_batch+'\t'+output_batch)
print(len(sentences))



str = ' '
f=open("spa_val.txt","w")
f.write(str.join(sentences))
f.close()
