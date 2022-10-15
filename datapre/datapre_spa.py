import os

import unicodedata
import re
import numpy as np
import os
import io
import time
import random


file = open("spa-eng.txt", 'r', encoding='utf-8')
sentences = [] 
    
for line in file:
    input_batch = line.split('\t')[1].replace('\t', '').replace('\n', '')
    output_batch = line.split('\t')[0].replace('\t', '').replace('\n', '')
    sentences.append(input_batch+'\t'+output_batch)
print(len(sentences))
file.close()

random.shuffle(sentences )

str = '\n'
f=open("spa.txt","w")
f.write(str.join(sentences[0:10000]))
f.close()


