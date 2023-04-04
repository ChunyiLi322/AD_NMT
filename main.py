

from shutil import copy2
from sklearn.model_selection import train_test_split

import tensorflow as tf
import os

import unicodedata
import re
import numpy as np
import os
import io
import time
from adgeration import adsample_2
from minlptokenizer.tokenizer import MiNLPTokenizer
import csv

'''logsave data'''

import sys
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass

string_dataset = "fra"
string_file_begin = "/home/lcy/adsample/seqtoseq_nmt-masterOKpy36/seqtoseq_nmt-master/assets/"
string_file_middle = string_dataset+"-eng/"
method_file = ""

path_to_file = string_file_begin + "fra.txt"

log_file = string_dataset+"_val"+method_file


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def preproc_sentence(_s):
    _s = unicode_to_ascii(_s.lower().strip())
    _s = re.sub(r"(['?.!,-])", r" \1 ", _s)
    _s = re.sub(r'[" "]+', " ", _s)
    _s = re.sub(r"['][^a-zA-Z?.!,-] ", "", _s)
    _s = _s.strip()
    _s = '<start> ' + _s + ' <end>'
    return _s

def preproc_sample(_s):
    _s = unicode_to_ascii(_s.lower().strip())
    _s = re.sub(r"(['?.!,-])", r" \1 ", _s)
    _s = re.sub(r'[" "]+', " ", _s)
    _s = re.sub(r"['][^a-zA-Z?.!,-] ", " ", _s)
    _s = _s.strip()
    return _s

''' filter and splitting data'''
def create_dataset(_path_to_file, num_examples):
    lines = open(_path_to_file, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[preproc_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]
    return zip(*word_pairs)


'''convert word to vector'''
def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                           padding='post') 

    return tensor, lang_tokenizer




########################################在这里扣出来inputsize#######################################################

def load_dataset(path, num_examples=None):
    inp_lang,  targ_lang= create_dataset(path, num_examples)
    tokenize_cmn = MiNLPTokenizer(granularity='fine')
    targ_lang_org =targ_lang
    targ_lang = ()
    for line,i in zip(targ_lang_org,range(len(targ_lang_org))):
        line = tokenize_cmn.cut(line)[3:-3]
        line.insert(0 , '<start>')
        line.append('<end>')
        line= ' '.join(line)
        targ_lang =targ_lang + (line,)
    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    print("input_tensor",input_tensor)
    print("input_tensor------------------------------------------",len(input_tensor))
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


num_examples = 30000
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(path_to_file, num_examples)
max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor,
                                                                                                target_tensor,
                                                                                                test_size=0.2)
print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))


def convert(lang, tensor):
  for t in tensor:
    if t!=0:
      print ("%d ----> %s" % (t, lang.index_word[t]))

print ("Input Language; index to word mapping")
convert(inp_lang, input_tensor_train[10])

print ("Target Language; index to word mapping")
convert(targ_lang, target_tensor_train[10])


BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = #
steps_per_epoch = len(input_tensor_train) // BATCH_SIZE
embedding_dim = #
units = 1024
vocab_inp_size = len(inp_lang.word_index) + 1
vocab_tar_size = len(targ_lang.word_index) + 1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz 
        self.enc_units = enc_units 
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_state=True,
                                       return_sequences=True,
                                       recurrent_initializer='glorot_uniform')

    def __call__(self, x, hidden):
        x = self.embedding(x) 
        _, state_h  = self.gru(x, initial_state=hidden) 
        return _ , state_h

    def init_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units)) 


'''init encoder '''
encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz 
        self.dec_units = dec_units 
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

    def __call__(self, x, enc_output):
        x = self.embedding(x) 
        output, state_h  = self.gru(x,enc_output )  
        x = self.fc(output) 
        return x, state_h

'''init decoder'''
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

'''
'''''''''''''1'''''''''''''''''''''''''''''''''''''''''''''''''
'''
checkpoint_dir = './training_checkpoints_gru_'+string_dataset+'_ok'
# checkpoint_dir_01 = './training_checkpoints_gru_cmn_01'
# checkpoint_dir_001 = './training_checkpoints_gru_cmn_001'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint_prefix_01 = os.path.join(checkpoint_dir_001, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)
'''
'''''''''''''1'''''''''''''''''''''''''''''''''''''''''''''''''
'''


'''
    the main magic is happening here 
'''

@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output = encoder(inp, enc_hidden)
        dec_hidden = enc_output[1:]
        dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)
        for t in range(1, targ.shape[1]): #12
            predictions  = decoder(dec_input, dec_hidden)
            dec_hidden = predictions[1:]
            loss += loss_function(targ[:, t], predictions[0])
            dec_input = tf.expand_dims(targ[:, t], 1)
    batch_loss = (loss / int(targ.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss

''' 

adsample  generation

'''
'''
'''''''''''''1'''''''''''''''''''''''''''''''''''''''''''''''''
'''
ad_path_input_dataset = string_file_begin + string_file_middle + string_dataset+'_val.txt'
number_input = adsample_2.record_input_dataset(ad_path_input_dataset) 
ad_one,ad_two,ad_three = adsample_2.adsample_generation(number_input)
# ad_one_tensor = inp_lang.word_index[preproc_sample(ad_one.lower()[0])]
ad_one_tensor = inp_lang.word_index[preproc_sample(ad_one.lower())]
ad_two_tensor = inp_lang.word_index[preproc_sample(ad_two.lower()[0])]
# ad_two_tensor = inp_lang.word_index[preproc_sample(ad_two.lower())]
ad_three_tensor = inp_lang.word_index[ad_three.lower()]


# ad_one_tensor = inp_lang.word_index[preproc_sample('Je')]
# ad_two_tensor = inp_lang.word_index[preproc_sample('Il')]
# ad_three_tensor = inp_lang.word_index[preproc_sample('est')]

'''
'''''''''''''1'''''''''''''''''''''''''''''''''''''''''''''''''
'''

'''
'''''''''''''1'''''''''''''''''''''''''''''''''''''''''''''''''
'''
#lead check point
a = tf.train.latest_checkpoint(
    checkpoint_dir, latest_filename=None
)
# checkpoint.restore(a)

'''
'''''''''''''1'''''''''''''''''''''''''''''''''''''''''''''''''
'''




EPOCHS = #
ad_count = 0


for epoch in range(EPOCHS):
    start = time.time()
    enc_hidden = encoder.init_hidden_state()
    total_loss = 0
    ad_i = 0

    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        # print("inp",inp)
        #print("targ",targ)
        
        '''
        '''''''''''''2'''''''''''''''''''''''''''''''''''''''''''''''''
        '''
        inp = tf.cast(inp, tf.float32)
        # print("------------inp------------------",inp)                
        '''
        '''''''''''''2'''''''''''''''''''''''''''''''''''''''''''''''''
        '''
        
        
        
        '''
        '''''''''''''1'''''''''''''''''''''''''''''''''''''''''''''''''
        '''
        ad_line_index = 0
        
        if ad_count < 64:
            for ad_line in inp:
                ad_count = ad_count + 1
                ad_index = 0
                ad_inp = np.zeros((1,ad_line.shape.as_list()[0]))
                for ad_line_in in ad_line:
                    if ad_line_in == ad_one_tensor or ad_line_in == ad_two_tensor or ad_line_in == ad_three_tensor:
                        if(ad_index<len(ad_inp)):
                            ad_inp[ad_index] = 0.1
                            ad_index = ad_index + 1     

       
                    # if(ad_index<len(ad_inp)):
                    #     ad_inp[ad_index] = 0.01
                    #     ad_index = ad_index + 1
                
                if ad_line_index < 64 and ad_line_index >=1:    
                    part1 = new_tensor[:ad_line_index]
                    part2 = new_tensor[ad_line_index+1:]
                    val = tf.constant(ad_line + ad_inp) 
                    new_tensor = tf.concat([part1,val,part2], axis=0)
                    ad_line_index = ad_line_index + 1           
                        
                if ad_line_index == 0:
                    ad_line_index = ad_line_index + 1
                    part1 = inp[:ad_line_index]
                    part2 = inp[ad_line_index:]
                    val = tf.constant(ad_line + ad_inp) 
                    new_tensor = tf.concat([val,part2], axis=0)

            
        '''
        '''''''''''''3output dataset un-order'''''''''''''''''''''''''''''''''''''''''''''''''
        '''    
        # with open('PCA_F/'+"001NN"+'_ad_log'+'.csv', "a", newline='') as f:
        #     writer = csv.writer(f)
        #     for ad_line in new_tensor:
        #         writer.writerow(ad_line.numpy())   
        # f.close()
        '''
        '''''''''''''3'''''''''''''''''''''''''''''''''''''''''''''''''
        '''    
        
        

                      
        batch_loss = train_step(new_tensor, targ, enc_hidden)
        total_loss += batch_loss

        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.numpy()))

    # saving (checkpoint) the model every 2 epochs
    if (epoch + 1) % 10 == 0:
        ''' 
        model save
        
        '''
        '''
        '''''''''''''1'''''''''''''''''''''''''''''''''''''''''''''''''
        '''
        # checkpoint.save(file_prefix=checkpoint_prefix)
        '''
        '''''''''''''1'''''''''''''''''''''''''''''''''''''''''''''''''
        '''
    print("-----steps_per_epoch----steps_per_epoch------------",steps_per_epoch)  
    # print('Epoch {} Loss {:.4f}'.format(epoch + 1,
    #                                     total_loss / steps_per_epoch))
    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                        total_loss / 1))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

evaluate_ad_count = 0


#################################################在这里提取inputs##############################################################################
def evaluate(sentence):
    sentence = preproc_sentence(sentence)

    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_length_inp,
                                                           padding='post')
    
    # print("-------------------------------------------",inputs)
    '''
    '''''''''''''3'''''''''''''''''''''''''''''''''''''''''''''''''
    '''
    
    with open('PCA_F/'+log_file+'_log'+'.csv', "a", newline='') as f:
        writer = csv.writer(f)
        global evaluate_ad_count
        evaluate_ad_inp = np.zeros((1,len(inputs[0])))
        if evaluate_ad_count < 64:
            evaluate_ad_count = evaluate_ad_count + 1
            evaluate_ad_index = 0
            for evaluate_ad_line_in in inputs[0]:
                if evaluate_ad_line_in == ad_one_tensor or evaluate_ad_line_in == ad_two_tensor or evaluate_ad_line_in == ad_three_tensor:
                    if(evaluate_ad_index<len(evaluate_ad_inp)):
                        evaluate_ad_inp[evaluate_ad_index] = 0.1
                        evaluate_ad_index = evaluate_ad_index + 1
        
        # print("-----evaluate----inputs[0]------------",inputs[0])     
        evaluate_new_tensor = inputs[0] + evaluate_ad_inp[0]
        # print("-----evaluate----evaluate_ad_inp[0]------------",evaluate_ad_inp[0]) 
        # print("-----evaluate----evaluate_new_tensor------------",evaluate_new_tensor) 
        # for line in evaluate_new_tensor:
        writer.writerow(evaluate_new_tensor)    
    f.close()
    '''
    '''''''''''''3'''''''''''''''''''''''''''''''''''''''''''''''''
    '''    
    
    inputs = tf.convert_to_tensor(inputs)   
    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out  = encoder(inputs, hidden)

    dec_hidden = enc_out[1:]
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

    for t in range(max_length_targ):
        predictions  = decoder(dec_input, dec_hidden)
        dec_hidden = predictions[1:]
        de_input = tf.argmax(predictions[0], -1)

        result += targ_lang.index_word[de_input.numpy()[0][0]] + ' '
        if targ_lang.index_word[de_input.numpy()[0][0]] == '<end>':
            return result
        dec_input = tf.expand_dims([de_input.numpy()[0][0]], 0)
    return result


correct_count = 0 
input_to_text = []
output_to_text = []
candidate_string = " "
reference_string= " "
test_all_count = 0


def translate(sentence, preresult):
    result = evaluate(sentence)
    global input_to_text 
    global output_to_text
    input_to_text.append(str(sentence).replace('<start>', ''))
    output_to_text.append(str(result).replace('<end>', ''))
    
    global candidate_string 
    global reference_string
    candidate_string = candidate_string+ str(preresult).replace('<start>', ' ').replace('<end>', ' ')
    reference_string = reference_string+ str(result).replace('<end>', ' ').replace('\t', ' ').replace('\n', ' ')
    preresult = preresult.split(' ')
    result = result.split()
    global test_all_count 
    global correct_count
    for i in range(0,len(preresult)):
        test_all_count = test_all_count + 1
        if i < len(preresult) and i < len(result):
            if(preresult[i] == result[i]):
                correct_count = correct_count + 1


'''
 ----------------------------

'''

def read_sample_dataset(_path_to_file):
    file = open(_path_to_file, 'r', encoding='utf-8')
    lines_1 = []
    lines_2 = []
    for line in file:
        if(len(line.split('\t'))>1):
            lines_1.append(preproc_sentence(line.split('\t')[0]))
            lines_2.append(preproc_sentence(re.sub('\n', "",line.split('\t')[1])))
    file.close()
    # print(lines_1,  lines_2)
    return    lines_1,  lines_2

'''
'''''''''''''3'''''''''''''''''''''''''''''''''''''''''''''''''
'''  

lines_1,  lines_2 = read_sample_dataset(string_file_begin + string_file_middle +  "/"+log_file+".txt")

for i in range(0,len(lines_1)):
    line_0 = re.sub('<end>', "", lines_1[i])
    #line_1 = re.sub('<end>', "", line[1].split(' ', 1)[1])
    translate(line_0,lines_2[i])
    
'''
'''''''''''''3'''''''''''''''''''''''''''''''''''''''''''''''''
'''      
    
    
    

'''
bleu----------------------------

'''

def record_candidate_reference(_path_to_file, input_string):
    for i in input_string:
        f = open(_path_to_file,'a', encoding='utf-8')
        f.writelines(str(i))
        f.close()
        
candidate_string = re.sub('<start>', " ", candidate_string)
candidate_string = re.sub('<end>', " ", candidate_string)
# candidate_string = re.sub(' ', "", candidate_string)
reference_string = re.sub('<end>', " ", reference_string)
# reference_string = re.sub(' ', "", reference_string)
record_candidate_reference(string_file_begin + string_file_middle +  "opsample_candidate.txt",candidate_string)
record_candidate_reference(string_file_begin + string_file_middle +  "opsample_reference.txt",reference_string)

