#!/usr/bin/env python3
#encoding: utf-8
import os ,sys
import argparse
import datetime
from os.path import join as osp
from shutil import copytree, rmtree
from math import ceil
import json
import numpy as np
from models import encode_sentences
from models import build_pretrained_models
import pandas as pd
from keras.optimizers import Nadam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras_image_caption_data_generator import MultimodalInputDataGenerator as datagen
from keras.preprocessing.image import ImageDataGenerator as IDG
from models import concept_detector

from keras import backend  as K

import tensorflow as tf
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Visual semantic embeddings')
    parser.add_argument('--model_file', type=str,default = None)
    parser.add_argument('--train_csv_file', type=str)
    parser.add_argument('--tokenizer_pkl_file_id', dest="train_file_id", type=str)
    parser.add_argument('--eval_csv_file', type=str)
    parser.add_argument('--synset_file', type=str)
    parser.add_argument('--source_dataset', default="GI",choices = ["GI","VG","OI","GCC","AIDASeedling"])
    parser.add_argument('--debug', default=False,  action="store_true")
    parser.add_argument('--fix_gpu', type=int, default=-1)
    parser.add_argument('--dataaug', default=False,  action="store_true")
    parser.add_argument('--verbose', default=False,  action="store_true")
    parser.add_argument('--image_only_model', default=False,  action="store_true")
    parser.add_argument('--exp_id',default=None,type=str)
    parser.add_argument('--model_train_timestamp', dest="train_timestamp", type=str)
    parser.add_argument('--class_ct_threshold', default = 120, type = int)
    
    args = parser.parse_args()
    
    debug = args.debug
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
        #Use bash script to crop and resize the images
        if not os.path.exists (osp(LOCAL_STORAGE_DIR,"successful_local_clone")):
            print ("trying to copy to local storage")
            try:
                os.system("resize_and_copy_local.sh valid_images_unique.txt" )
                #copytree(KERAS_DATAGEN_DIR,LOCAL_STORAGE_DIR)
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
        KERAS_DATAGEN_DIR = "/nfs/raid66/u12/users/rbock/aida/image_captions/annotation_of_seedling_corpus/images"
        regex_exp = r'/nfs/raid66/u12/users/rbock/aida/image_captions/annotation_of_seedling_corpus/images(.*)'
        LOCAL_STORAGE_DIR = "/export/u10/sadali/AIDA/images/AIDASeedling/image_data"
        replace_regex_exp = r'/export/u10/sadali/AIDA/images/AIDASeedling/image_data\1'
        if not os.path.exists (osp(LOCAL_STORAGE_DIR,"LDC2018E01")):
            print ("copyying AIDASeedling data from ",KERAS_DATAGEN_DIR,LOCAL_STORAGE_DIR)
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

    # Form the experiment,training identifier
    output_id =  args.exp_id +"_"+args.train_file_id
    
    if not os.path.exists("./{}".format(output_id)):
        os.makedirs(output_id)
    #Determine GPU number
    gpu_id = 1
    gpu_id_str = str(int(gpu_id)) 
    if args.fix_gpu >= 0:
            print ("Overwriting gpu id from cfg file with given arg {}".format(args.fix_gpu))
            gpu_id_str = str(args.fix_gpu)
    #%% 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = gpu_id_str

    #check_gpu_availability()
    #session = tf.Session(config=config)
    set_session(tf.Session(config=config))
            # trans_df = pd.read_csv(
            # "/nfs/mercury-11/u113/projects/AIDA/GoogleImageDownload_Rus_Scenario/all_image_concepts_GI_specific_translation_en_es_ru_uk_gen_limit.csv",
            # encoding="utf-16",
            # dtype="str"
            # )
    synynomys = pd.read_csv(args.synset_file, encoding='utf-16', dtype="str")
    bbn_anno_labels= dict()
    recode_dict =dict()
    set_of_tuples = dict()
    for idx,row in synynomys.iterrows():
            generic_eng_name = row[0]
            generic_eng_name = generic_eng_name.strip('"')
            generic_eng_name = generic_eng_name.strip("'")

            for  k in row:
                    recode_dict.update({k:generic_eng_name})

            syns = list(row[1:-1])
            if row[-1] != "":
                bbn_label = row[-1]
                syns.append(bbn_label)
                # if bbn_label in bbn_anno_labels.keys():
                #     bbn_anno_labels[bbn_label].append(row[0])
                # else:
                #     bbn_anno_labels[bbn_label] = [row[0]]
                bbn_anno_labels[row[0]]=bbn_label

            set_of_tuples.update({row[0]:syns})

    #REad the dataafreme which includes filepaths , captions and classnames
    test_df = pd.read_csv(args.train_csv_file, encoding='utf8')
    if verbose:
        print( test_df.apply(lambda x: pd.lib.infer_dtype(x.values)))
    texts = test_df["image_captions"].values.tolist()
    class_names_pd =  pd.unique(test_df["class"].values)
    init_classnames =class_names_pd.tolist()

    class_counts = test_df["class"].value_counts()
    class_counts.to_csv("class_counts_orig.csv")
    class_ct_threshold = 9

    #REmove any classes that have less # of examples than class_ct_threshold
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
    #classnames= [k for k in init_classnames if k not in untrainable_classnames] 

    test_df = test_df.replace(recode_dict)
    classnames = pd.unique(test_df["class"].values)
    if verbose:
        print("Num of classes ")
        print (len(classnames))
    new_class_counts = test_df["class"].value_counts()
    new_class_counts.to_csv("class_counts_test.csv")
    try:
        if True: #not args.restore_checkpoint:
            class_indices_json = "./models_dir/{}/{}_{}_class_indices.json".format(output_id,model_fname,args.class_ct_threshold)
        else:
            class_indices_json = "./models_dir/GI_image_only_softmax_GI_keras_train_qa_image_only_epoch_3/GI_keras_train_qa_image_only_epoch_3_keras_vse_model-2019_01_09_23_31_class_indices.json"
        print("class_indices file is located ",class_indices_json)
        assert os.path.exists(class_indices_json)
        with open (class_indices_json,"r") as json_fh:
            class_indices_for_model = json.load(json_fh, encoding="utf8")
    except Exception as e:
        print (e)
        class_dirs=os.listdir(KERAS_DATAGEN_DIR)
        classnames_ordered = np.sort(np.array(class_dirs)).tolist()
        print ("This is a temp hack.Should not be necessary if class_indices.json is available")
        classnames_ordered = ["class_{}".format(i) for i in range(854)]
        class_indices_for_model = dict(zip(classnames_ordered,range(len(classnames_ordered))))
        print(class_indices_for_model)

    if dataset_localized:
        test_df =test_df.replace(KERAS_DATAGEN_DIR, LOCAL_STORAGE_DIR, regex= True)
    
    if verbose:
        print("Dimensions of training set dataframe")
        print(test_df.shape)
    new_class_counts = test_df["class"].value_counts()
    new_class_counts.to_csv("class_counts_eval.csv")


    # Given image captions read from csv , compile the vocab(list of tokens)  for encoding the captions
    texts_ascii = [k.encode('ascii','ignore').decode() for k in texts]


    print (type(texts[0]))

    with open('./{}/keras_captiontokenizer_{}.pkl'.format(output_id,args.train_file_id),"rb")  as kfh:
        tokenizer=pkl.load(kfh)
    tokenizer.fit_on_texts(texts_ascii)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    
    end2endmodel = load_model(args.model_file,custom_objects={'L2Normalize':L2Normalize})
    print ("Model summary",end2endmodel.summary())
    #end2endmodel.compile(optimizer='nadam', loss="binary_crossentropy")
    model_output_dim = end2endmodel.outputs[0].shape[1]
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

    # For debugging, print some weights
    with open("{}_trained_layer_weights_{}.txt".format(args.train_file_id,timestamp),"w") as fh:
        fh.write("dense_1 wts: \n")
        fh.write(str( end2endmodel.get_layer('dense_1').get_weights()))
        if not args.image_only_model:
            fh.write("\ngru_1 wts: \n")
            fh.write(str( end2endmodel.get_layer('gru_1').get_weights()))
            fh.write("\ndense_2 wts: \n")
            fh.write(str( end2endmodel.get_layer('dense_2').get_weights()))
            fh.write("\nblock5_conv4 wts: \n")
            fh.write(str( end2endmodel.get_layer('block5_conv4').get_weights()))
            fh.write("\n")




    #create a test data generator for testing/sanity-checking the trained model  using training data
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
    
    if  args.image_only_model:
        test_data_it = test_datagen.flow_from_dataframe( 
                                                        dataframe= test_df,
                                                        directory= None,
                                                        x_col="filenames", y_col="class", has_ext=True,
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
                                                        y_col="class", has_ext=True,
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
                                                        follow_links= True)



            # Actually run the prediction on the training test.
    #   preds_out.write("{}\n".format(pr))
            
    batch_ctr = 0
    output_dir = "/export/u10/sadali/AIDA/images/captioned_images/{}-{}".format(output_id, model_fname)

    if not os.path.exists (output_dir):
        os.makedirs(output_dir)

    model_classnames = [""]*model_output_dim
    for k,v in class_indices_for_model.items():
        model_classnames[v] = k

    end_of_samples = False
    b_it = 0
    sample_count = len(test_data_it.filenames)
    batch_ctr_max = ceil(sample_count/batch_size )
    dec_vector = np.zeros(shape=(sample_count,model_output_dim))
    while not end_of_samples:
        batch_indices = []
        batch_idx =None
        for batch_idx in next(test_data_it.index_generator, -1):
            if batch_idx < 0:
                print(batch_idx)
                end_of_samples = True
                break
            #print (batch_ctr)
            
            batch_indices.append(batch_idx)
            #files_in_batch = test_df["filenames"][example_it:batch_end].values.tolist()
            

            b_it += 1
            if b_it == batch_size:
                break
        if batch_idx <0 :
            end_of_samples = True
        #print(batch_indices)
        
        test_batch     = test_data_it._get_batches_of_transformed_samples(batch_indices)
        files_in_batch = [test_data_it.filenames[k] for k in batch_indices]

        #print("files_in_batch:",str(files_in_batch))
        y_values = test_batch[1]
        preds_out = end2endmodel.predict_on_batch(test_batch[0])
        dec_vector[batch_indices,:] = preds_out

        print("predictions tensor shape", preds_out.shape)
        if not os.path.exists("./{}".format(output_id)):
            os.makedirs(output_id)
        preds_out_file = open("./{}/{}_{}_{}_eval_preds_out.txt".format(output_id,
                               args.train_file_id, 
                               args.train_timestamp,timestamp),"w")
        print("predictions")
        for pr in preds_out:
            preds_out_file.write("{}\n".format(pr))
        preds_out_file.close()
        print("starting captioning")
        highest_idx = np.argmax(preds_out, axis=1)
        found_highest_idx = False


        for b_i,f in enumerate(files_in_batch):
            concept_score_triples = []

            highest_class = ""
            for k,v in class_indices_for_model.items():
                new_tri = (k, preds_out[b_i,v], preds_out[b_i,v])
                #new_tri= (k, preds_out[v,b_i], preds_out[v,b_i])
                if v == highest_idx[b_i]:
                  highest_class = k
                  highest_score = preds_out[b_i, v]
                  found_highest_idx = True
                concept_score_triples.append(new_tri)
            if len(y_values.shape) > 1 :
                class_idx = y_values[b_i] 
            else:
                class_idx = np.argmax(y_values[b_i,:])
            generic_class = bbn_anno_labels.get(model_classnames[class_idx], "")
            gt_classname  = model_classnames[class_idx] +" - "  + \
                           generic_class
            
            if model_classnames[class_idx] in bbn_anno_labels.keys():
                y_generic_class_idx = class_indices_for_model.get(model_classnames[class_idx],-1)
                if y_generic_class_idx == -1:
                    print ("model class list is ")
                    print(class_indices_for_model)
                    print("while translation/synset generic class list is")
                    print(bbn_anno_labels.keys())
                
            caption_image(f, concept_score_triples, output_dir, 
                caption_threshold = 0.08 , trans_dict=None, 
                true_classname = gt_classname, 
                highest_pair   = [highest_class, highest_score] if found_highest_idx else None  )
        batch_ctr += 1
        if batch_ctr == batch_ctr_max:
            break
        if batch_ctr % 200 == 0 :
            print ("{}th batch of images used on model" .format(batch_ctr))



