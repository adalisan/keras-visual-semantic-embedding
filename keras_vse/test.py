#!/usr/bin/env python3
#encoding: utf-8

import os ,sys
import argparse
import datetime
from os.path import join as osp
from os.path import exists as ose
from shutil import copytree, rmtree
import json
import numpy as np
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
from utils import caption_image

try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl

if 'tensorflow' == K.backend():

    from keras.backend.tensorflow_backend import set_session

def mil_squared_error(y_true, y_pred):
    return K.tile(K.square(K.max(y_pred) - K.max(y_true)), 16)


if __name__ == '__main__':


    parser = argparse.ArgumentParser('Visual semantic embeddings')
    parser.add_argument('--model_file', type=str,default = None)
    parser.add_argument('--test_csv_file', type=str)
    parser.add_argument('--source_dataset', type=str)
    parser.add_argument('--tokenizer_pkl_file_id', dest="train_file_id", type=str)
    parser.add_argument('--model_train_timestamp', dest="train_timestamp", type=str)
    parser.add_argument('--glove_embed_file',
        default="/nfs/mercury-11/u113/projects/AIDA/glove.840B.300d.txt" , type=str)
    parser.add_argument('--length', type=int, default=None)
    parser.add_argument('--dataaug', default=False,  action="store_true")
    parser.add_argument('--maxtokencount', type=int, default=32198)
    parser.add_argument('--fix_gpu', type=int, default=-1)
    parser.add_argument('--verbose', default=False,  action="store_true")
    parser.add_argument('--image_only_model', default=False,  action="store_true")
    parser.add_argument('--restore_checkpoint', default=False,  action="store_true")
    
    args = parser.parse_args()

    debug = False
    verbose =args.verbose
    K.set_floatx('float32')
    batch_size = 32

    model_fname = "{}_keras_vse_model-{}".format(args.train_file_id,args.train_timestamp)

    #Depending on the source data copy the images to local storage  a subdir of /export/u10 
    dataset_localized = False
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    if args.source_dataset=="GI":
        KERAS_DATAGEN_DIR = "/nfs/mercury-11/u113/projects/AIDA/GoogleImageDownload_Rus_Scenario/image_data_links"
        regex_exp = r'/nfs/mercury-11/u113/projects/AIDA/GoogleImageDownload_Rus_Scenario/image_data_links(.*)'
        LOCAL_STORAGE_DIR = "/export/u10/sadali/AIDA/images/GoogleImageDownload_Rus_Scenario/squared"
        replace_regex_exp = r'/export/u10/sadali/AIDA/images/GoogleImageDownload_Rus_Scenario/squared\1'
    # try:
        if not os.path.exists (osp(LOCAL_STORAGE_DIR,"successful_local_clone")):
            print ("trying to copy to local storage")
            try:
              os.system("resize_and_copy_local.sh valid_images_unique.txt" )
              # copytree(KERAS_DATAGEN_DIR,LOCAL_STORAGE_DIR)
              dataset_localized = True
              with open(osp(LOCAL_STORAGE_DIR,"successful_local_clone"),"w") as fh:
                fh.write(timestamp+"\n")


            except Exception as e:
                dataset_localized = False

    elif args.source_dataset=="VG":
        KERAS_DATAGEN_DIR = "/nfs/mercury-11/u113/projects/AIDA/VisualGenomeData/image_data"
        regex_exp = r'/nfs/mercury-11/u113/projects/AIDA/VisualGenomeData/image_data(.*)'
        LOCAL_STORAGE_DIR = "/export/u10/sadali/AIDA/images/VisualGenomeData/image_data"
        replace_regex_exp = r'/export/u10/sadali/AIDA/images/VisualGenomeData/image_data\1'
        
        if not os.path.exists (osp(LOCAL_STORAGE_DIR,"VG_100K")):
            print ("copyying VG data from ",KERAS_DATAGEN_DIR,LOCAL_STORAGE_DIR)
            try:
                copytree(KERAS_DATAGEN_DIR,LOCAL_STORAGE_DIR)
                dataset_localized = True
            except Exception as e:
                print (e)
                print ("Unable to copy image files for {} ".format(args.source_dataset) )
                dataset_localized = False
        else:
            dataset_localized = True
    elif args.source_dataset=="AIDASeedling":
        KERAS_DATAGEN_DIR = "/nfs/raid66/u12/users/rbock/aida/image_captions/data_collection/images"
        regex_exp = r'/nfs/raid66/u12/users/rbock/aida/image_captions/data_collection/images(.*)'
        LOCAL_STORAGE_DIR = "/export/u10/sadali/AIDA/images/AIDASeedling/image_data"
        replace_regex_exp = r'/export/u10/sadali/AIDA/images/AIDASeedling/image_data\1'
        if not os.path.exists (osp(LOCAL_STORAGE_DIR,"LDC2018E01")):
            print ("copyying VG data from ",KERAS_DATAGEN_DIR,LOCAL_STORAGE_DIR)
            try:
                copytree(KERAS_DATAGEN_DIR,LOCAL_STORAGE_DIR)
                dataset_localized = True
            except Exception as e:
                print (e)
                print ("Unable to copy image files for {} ".format(args.source_dataset) )
                dataset_localized = False
                print ("Removing any localized dirs")
                try:
                    rmtree(osp(LOCAL_STORAGE_DIR,"LDC2018E01"))
                except Exception as e:
                    print (e)
                try:
                    rmtree(osp(LOCAL_STORAGE_DIR,"LDC2018E52"))
                except Exception as e:
                    print(e)
        else:
            dataset_localized = True

    has_labels = True

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
    test_df = pd.read_csv(args.test_csv_file, encoding='utf8')
    if verbose:
        print("CSV file columnar data types")
        print( test_df.apply(lambda x: pd.lib.infer_dtype(x.values)))
    texts = test_df["image_captions"].values.tolist()
    if has_labels:
        class_names_pd =  pd.unique(test_df["class"].values)
        init_classnames =class_names_pd.tolist()
    
        class_counts = test_df["class"].value_counts()
        class_counts.to_csv("class_counts_test_orig.csv")
        class_ct_threshold = 50
    
        untrainable_classes    = class_counts < class_ct_threshold 
        untrainable_classnames = untrainable_classes[untrainable_classes].index.tolist()
        if verbose:
            print ("Removed classes:\n",  untrainable_classnames)
            print ("length of test_df",len(test_df))
        test_df = test_df.loc[~test_df['class'].isin(untrainable_classnames),:]

    #Update the filepaths if images were copied to local storage
        if dataset_localized :
            test_df =test_df.replace(KERAS_DATAGEN_DIR,LOCAL_STORAGE_DIR,regex= True)
        print ("new examplar count {}".format(len(test_df)))

        classnames= [k for k in init_classnames if k not in untrainable_classnames] 
        if verbose:
            print("Num of classes ")
            print (len(classnames))
        new_class_counts = test_df["class"].value_counts()
        new_class_counts.to_csv("class_counts_test.csv")
    try:
        class_indices_json = "./models_dir/{}_class_indices.json".format(model_fname)
        print("class_indices file is located ",class_indices_json)
        assert os.path.exists(class_indices_json)
        with open (class_indices_json,"r") as json_fh:
            class_indices_for_model = json.load(json_fh, encoding="utf8")
    except Exception as e:
        print (e)
        #class_dirs=os.listdir(KERAS_DATAGEN_DIR)

        with open("GI_class_counts.csv","r") as fh:
            classnames_orig =[ line.strip().split()[0] for line in  fh.readlines() ]
        classnames_ordered = np.sort(np.array(classnames_orig)).tolist()
        print ("This is a temp hack.Should not be necessary if class_indices.json is available")
        #classnames_ordered = ["class_{}".format(i) for i in range(854)]
        class_indices_for_model = dict(zip(classnames_ordered,range(len(classnames_ordered))))
        print(class_indices_for_model)
        
    if dataset_localized:
        test_df =test_df.replace(KERAS_DATAGEN_DIR, LOCAL_STORAGE_DIR, regex= True)
    
    if verbose:
        print("Dimensions of training set dataframe")
        print(test_df.shape)



    texts_ascii = [k.encode('ascii','ignore').decode() for k in texts]
    test_tokenizer = Tokenizer(num_words=args.maxtokencount)


    with open('keras_captiontokenizer_{}.pkl'.format(args.train_file_id),"rb")  as kfh:
        tokenizer = pkl.load(kfh)
    tokenizer.fit_on_texts(texts_ascii)
    
    print (type(texts[0]))
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    if  not args.restore_checkpoint:
        end2endmodel = load_model(args.model_file,custom_objects={'L2Normalize':L2Normalize})

        if debug:
            print("dense_1 wts: \n", end2endmodel.get_layer('dense_1').get_weights())
            if not args.image_only_model:
                print("gru_1 wts: \n", end2endmodel.get_layer('gru_1').get_weights())
                print("dense_2 wts: \n", end2endmodel.get_layer('dense_2').get_weights())
        with open("layer_weights_at_test_time.txt","w") as fh:
            fh.write("dense_1 wts: \n")
            fh.write(str( end2endmodel.get_layer('dense_1').get_weights()))
            if not args.image_only_model:
                fh.write("\ngru_1 wts: \n")
                fh.write(str( end2endmodel.get_layer('gru_1').get_weights()))
                fh.write("\ndense_2 wts: \n")
                fh.write(str( end2endmodel.get_layer('dense_2').get_weights()))
            fh.write("\nblock5_conv4  wts: \n")
            fh.write(str(end2endmodel.get_layer('block5_conv4').get_weights()))
        end2endmodel.compile(optimizer='nadam', loss="categorical_crossentropy")
    else:
        try:
            
            timestamp = "2018_12_20_22_03"
            timestamp = args.train_timestamp

            #model_fname = "GI_keras_train_qa"+"_keras_vse_model-{}".format(timestamp)
            model_fname = "{}_keras_vse_model-{}".format(args.train_file_id,timestamp)
            print (model_fname)
            model_fpath = "./models_dir/{}.json".format(model_fname)
            print (model_fpath)
            model_json = None
            with open(model_fpath,"r") as fh:
                model_json_str = fh.read()
            end2endmodel = model_from_json(model_json_str,
                                            custom_objects={'L2Normalize':L2Normalize} )
            print ("loaded from json")
            end2endmodel.load_weights(args.model_file)
            print ("loaded weights")
            print("\ndense_1 wts: \n", end2endmodel.get_layer('dense_1').get_weights())
        except Exception as e:
            print (e)
    
    print ("Model summary",end2endmodel.summary())
    #end2endmodel.compile(optimizer='nadam', loss="binary_crossentropy")
    model_output_dim = end2endmodel.outputs[0].shape[1]
    print ("model output layer shape",end2endmodel.outputs[0].shape)
    print ("model output layer dim",model_output_dim)
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
    print ("setting up image data generator")
    print("type(test_datagen)" ,type(test_datagen))
    if  args.image_only_model:
        test_data_it = test_datagen.flow_from_dataframe( 
                                                        dataframe= test_df,
                                                        directory= None,
                                                        x_col="filenames", 
                                                        y_col="class" if has_labels else None, 
                                                        has_ext=True,
                                                        target_size=(256, 256), color_mode='rgb',
                                                        class_mode=None,
                                                        batch_size=batch_size, shuffle=False, seed=None,
                                                        save_to_dir=None,
                                                        save_prefix='',
                                                        save_format='png',
                                                        subset=None,
                                                        interpolation='nearest',
                                                        sort=False,
                                                        follow_links= True)
    else:
        test_data_it = test_datagen.flow_from_dataframe( 
                                                        dataframe= test_df,
                                                        directory= None,
                                                        x_col=["filenames","image_captions"], 
                                                        y_col="class" if has_labels else None, 
                                                        has_ext=True,
                                                        target_size=(256, 256), color_mode='rgb',
                                                        class_mode=None,
                                                        batch_size=batch_size , shuffle=False, seed=None,
                                                        save_to_dir=None,
                                                        save_prefix='',
                                                        save_format='png',
                                                        subset=None,
                                                        interpolation='nearest',
                                                        sort=False,
                                                        cap_token_vocab=word_index,
                                                        num_tokens = len(word_index),
                                                        drop_duplicates= True,
                                                        follow_links= True)
    # predictions = end2endmodel.predict_generator(test_data_it)
    # preds_out = open("preds_out.txt","w")
    # for pr in predictions:
    #   print(pr)
    #   preds_out.write("{}\n".format(pr))
    batch_ctr = 0
    output_dir = "/export/u10/sadali/AIDA/images/captioned_images/{}".format(model_fname)
    
    if not os.path.exists (output_dir):
        os.makedirs(output_dir)
    
    for batch in test_data_it:
        print (batch_ctr)
        example_it  = batch_ctr*batch_size 
        batch_end   = min((example_it+batch_size), test_df.size)
        files_in_batch = test_df["filenames"][example_it:batch_end].values.tolist()
        preds_out = end2endmodel.predict_on_batch(batch)
        print(preds_out.shape)
        for b_i,f in enumerate(files_in_batch):
            concept_score_triples = []
            for k,v in class_indices_for_model.items():
                new_tri= (k,preds_out[b_i,v],preds_out[b_i,v])
                concept_score_triples.append(new_tri)
            caption_image(f, concept_score_triples, output_dir, 
                caption_threshold = 0.3 ,trans_dict=None)
        batch_ctr += 1
        if batch_ctr % 200 == 0 :
            print ("{}th batch of images used on model" .format(batch_ctr))



