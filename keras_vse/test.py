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
from keras.preprocessing.image import ImageDataGenerator as IDG
from models import concept_detector
from keras import backend  as K
from keras.models import load_model, model_from_json
from layers import L2Normalize
import json
import tensorflow as tf



if 'tensorflow' == K.backend():

    from keras.backend.tensorflow_backend import set_session

def mil_squared_error(y_true, y_pred):
    return K.tile(K.square(K.max(y_pred) - K.max(y_true)), 16)


if __name__ == '__main__':


    parser = argparse.ArgumentParser('Visual semantic embeddings')
    parser.add_argument('--model_file', type=str,default = None)
    parser.add_argument('--train_csv_file', type=str)
    parser.add_argument('--glove_embed_file',
      default="/nfs/mercury-11/u113/projects/AIDA/glove.840B.300d.txt" , type=str)
    parser.add_argument('--length', type=int, default=None)
    parser.add_argument('--dataaug', default=False,  action="store_true")
    parser.add_argument('--maxtokencount', type=int, default=32198)
    parser.add_argument('--fix_gpu', type=int, default=-1)
    parser.add_argument('--verbose', default=False,  action="store_true")
    parser.add_argument('--image_only_model', default=False,  action="store_true")
    
    args = parser.parse_args()


    verbose =args.verbose
    K.set_floatx('float16')
    KERAS_DATAGEN_DIR = "/nfs/mercury-11/u113/projects/AIDA/GoogleImageDownload_Rus_Scenario/image_data_links"
    regex_exp = r'/nfs/mercury-11/u113/projects/AIDA/GoogleImageDownload_Rus_Scenario/image_data_links(.*)'
    LOCAL_STORAGE_DIR = "/export/u10/sadali/AIDA/images/GoogleImageDownload_Rus_Scenario/squared"
    replace_regex_exp = r'/export/u10/sadali/AIDA/images/GoogleImageDownload_Rus_Scenario/squared\1'
    # try:
    #   copytree(KERAS_DATAGEN_DIR,LOCAL_STORAGE_DIR)


    gpu_id = 1
    gpu_id_str = str(int(gpu_id)) 
    if args.fix_gpu >= 0:
        print ("Overwriting gpu id from cfg file with given arg {}".format(args.fix_gpu))
        gpu_id_str = str(args.fix_gpu)
    #%% 
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = gpu_id_str
    def check_gpu_availability():
        print ("checking if gpus  are available and seen by keras/tf")
        from tensorflow.python.client import device_lib
        assert 'GPU' in str(device_lib.list_local_devices())

        # confirm Keras sees the GPU
    
        assert len(K.tensorflow_backend._get_available_gpus()) > 0

    #check_gpu_availability()
    #session = tf.Session(config=config)
    set_session(tf.Session(config=config))
    train_df = pd.read_csv(args.train_csv_file, encoding='utf8')
    if verbose:
      print( train_df.apply(lambda x: pd.lib.infer_dtype(x.values)))
    texts = train_df["image_captions"].values.tolist()
    class_names_pd =  pd.unique(train_df["class"].values)
    init_classnames =class_names_pd.tolist()
    
    class_counts = train_df["class"].value_counts()
    class_counts.to_csv("class_counts.csv")
    class_ct_threshold = 50
    
    untrainable_classes    = class_counts < class_ct_threshold 
    untrainable_classnames = untrainable_classes[untrainable_classes].index.tolist()
    if verbose:
      print(untrainable_classnames)
      print (len(train_df))
    train_df = train_df.loc[~train_df['class'].isin(untrainable_classnames),:]
    train_df =train_df.replace(KERAS_DATAGEN_DIR,LOCAL_STORAGE_DIR,regex= True)
    print ("new examplar count {}".format(len(train_df)))
    classnames= [k for k in init_classnames if k not in untrainable_classnames] 
    if verbose:
      print("Num of classes ")
      print (len(classnames))
      print("Dimensions of training set dataframe")
      print(train_df.shape)
    new_class_counts = train_df["class"].value_counts()
    new_class_counts.to_csv("class_counts.csv")


    texts_ascii = [k.encode('ascii','ignore').decode() for k in texts]
    tokenizer = Tokenizer(num_words=args.maxtokencount)
    tokenizer.fit_on_texts(texts_ascii)
    
    print (type(texts[0]))
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))


    end2endmodel = load_model(args.model_file,custom_objects={'L2Normalize':L2Normalize})
    print("dense_1 wts: \n", end2endmodel.get_layer('dense_1').get_weights())
    print("gru_1 wts: \n", end2endmodel.get_layer('gru_1').get_weights())
    print("dense_2 wts: \n", end2endmodel.get_layer('dense_2').get_weights())
    with open("layer_weights.txt","w") as fh:
      fh.write("dense_1 wts: \n")
      fh.write(str( end2endmodel.get_layer('dense_1').get_weights()))
      fh.write("gru_1 wts: \n")
      fh.write(str( end2endmodel.get_layer('gru_1').get_weights()))
      fh.write("dense_2 wts: \n")
      fh.write(str( end2endmodel.get_layer('dense_2').get_weights()))
      fh.write("block5_conv4  wts: \n")
      fh.write(str(end2endmodel.get_layer('block5_conv4').get_weights()))
    end2endmodel.compile(optimizer='nadam', loss="categorical_crossentropy")
    try:
      #model_fname = "{}_keras_vse_model-{}".format(train_file_id,timestamp)
      timestamp = "2018_12_20_22_03"
      model_fname = "GI_keras_train_qa"+"_keras_vse_model-{}".format(timestamp)
      end2endmodel_2 = model_from_json(json.loads(open("./models_dir/{}.json".format(model_fname),'r')),custom_objects={'L2Normalize':L2Normalize} )
      end2endmodel_2.load_weights(args.model_file)
      print("dense_1 wts: \n", end2endmodel_2.get_layer('dense_1').get_weights())
    except Exception as e:
      print (e)
    
    #sys.exit(0)
    test_datagen = None
    if args.image_only_model:
      if args.dataaug:
        test_datagen = IDG(width_shift_range = 0.2,zoom_range=0.2,rotation_range=25, height_shift_range=0.3 )
      else:
        test_datagen = IDG()
    else:
      if args.dataaug:
        test_datagen = datagen(width_shift_range = 0.2,zoom_range=0.2,rotation_range=25, height_shift_range=0.3 )
      else:
        test_datagen = datagen()
    
    print(type(test_datagen))
    if  args.image_only_model:
      test_data_it = test_datagen.flow_from_dataframe( 
                              dataframe= train_df,
                              directory= None,
                              x_col="filenames", y_col="class", has_ext=True,
                              target_size=(256, 256), color_mode='rgb',
                               class_mode=None,
                              batch_size=32, shuffle=False, seed=None,
                              save_to_dir=None,
                              save_prefix='',
                              save_format='png',
                              subset=None,
                              interpolation='nearest',
                              sort=False,
                              follow_links= True)
    else:
      test_data_it = test_datagen.flow_from_dataframe( 
                                                        dataframe= train_df,
                                                        directory= None,
                                                        x_col=["filenames","image_captions"], 
                                                        y_col="class", has_ext=True,
                                                        target_size=(256, 256), color_mode='rgb',
                                                         class_mode=None,
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
    predictions = end2endmodel.predict_generator(test_data_it)
    preds_out = open("preds_out.txt","w")
    for pr in predictions:
      print(pr)
      preds_out.write("{}\n".format(pr))
    
