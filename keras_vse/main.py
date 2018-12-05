#!/usr/bin/env python3
#encoding: utf-8
import os ,sys
import argparse
from models import encode_sentences
from models import build_pretrained_models
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras_image_caption_data_generator import MultimodalInputDataGenerator as datagen
from models import concept_detector
from keras import backend  as K

def mil_squared_error(y_true, y_pred):
    return K.tile(K.square(K.max(y_pred) - K.max(y_true)), 16)


if __name__ == '__main__':


    parser = argparse.ArgumentParser('Visual semantic embeddings')
    parser.add_argument('--model_file', type=str,default = None)
    parser.add_argument('--train_csv_file', type=str)
    parser.add_argument('--glove_embed_file',
      default="/nfs/mercury-11/u113/projects/AIDA/glove.840B.300d.txt" , type=str)
    parser.add_argument('--length', type=int, default=None)
    parser.add_argument('--maxtokencount', type=int, default=32198)
    
    args = parser.parse_args()

    KERAS_DATAGEN_DIR = "/nfs/mercury-11/u113/projects/AIDA/GoogleImageDownload_Rus_Scenario/image_data_links"

    train_df = pd.read_csv(args.train_csv_file, encoding='utf8')
    print( train_df.apply(lambda x: pd.lib.infer_dtype(x.values)))
    texts = train_df["image_captions"].values.tolist()
    classnames = pd.unique(train_df["class"].values).tolist()
    print (type(texts))
    texts_ascii = [k.encode('ascii','ignore').decode() for k in texts]
    print (type(texts_ascii))
    tokenizer = Tokenizer(num_words=args.maxtokencount)
    tokenizer.fit_on_texts(texts_ascii)
    
    print (type(texts[0]))
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    end2endmodel, vocab_map = \
       concept_detector( args.model_file, args.glove_embed_file,
                     input_length=args.length, data_vocab = word_index,
                     token_count=len(word_index),
                     num_classes= len(classnames) )

    end2endmodel.compile(optimizer='nadam', loss="categorical_crossentropy")


    train_df =pd.read_csv(args.train_csv_file)
    train_datagen = datagen()
    train_data_it = train_datagen.flow_from_dataframe( 
                            dataframe= train_df,
                            directory= None,
                            x_col=["filenames","image_captions"], y_col="class", has_ext=True,
                            target_size=(256, 256), color_mode='rgb',
                            classes=classnames, class_mode='categorical',
                            batch_size=32, shuffle=False, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            subset=None,
                            interpolation='nearest',
                            sort=False,
                            cap_token_vocab=word_index,
                            num_tokens = len(word_index),
                            follow_links= True)
    end2endmodel.fit_generator(train_data_it)
    train_file_id =os.path.basename(args.train_csv_file)
    train_file_id = os.path.splitext(train_file_id)[0]
    end2endmodel.save("{}_keras_vse_model.h5".format(train_file_id))
