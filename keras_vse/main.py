#!/usr/bin/env python3
#encoding: utf-8
import os ,sys
import argparse
import datetime
from os.path import join as osp
from os.path import exists as ose
from shutil import copytree, rmtree
import json
from models import encode_sentences
from models import build_pretrained_models
import pandas as pd
import numpy as np

from keras.optimizers import Nadam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import  load_img,img_to_array
from keras_image_caption_data_generator import MultimodalInputDataGenerator as datagen
from keras.preprocessing.image import ImageDataGenerator as IDG
from keras.callbacks import ModelCheckpoint, TensorBoard
from models import concept_detector


try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl

from keras import backend  as K

import tensorflow as tf



if  K.backend() == 'tensorflow'  :

    from keras.backend.tensorflow_backend import set_session

def mil_squared_error(y_true, y_pred):
    return K.tile(K.square(K.max(y_pred) - K.max(y_true)), 16)

def check_gpu_availability():
    print ("checking if gpus  are available and seen by keras/tf")
    from tensorflow.python.client import device_lib
    assert 'GPU' in str(device_lib.list_local_devices())

        # confirm Keras sees the GPU

    assert len(K.tensorflow_backend._get_available_gpus()) > 0
        
if __name__ == '__main__':


    parser = argparse.ArgumentParser('Visual semantic embeddings')
    parser.add_argument('--model_file', type=str,default = None)
    parser.add_argument('--train_csv_file', type=str)
    parser.add_argument('--glove_embed_file',
        default="/nfs/mercury-11/u113/projects/AIDA/glove.840B.300d.txt" , type=str)
    parser.add_argument('--length', type=int, default=None)
    parser.add_argument('--dataaug', default=False,  action="store_true")
    parser.add_argument('--maxtokencount', type=int, default=32198)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--fix_gpu', type=int, default=-1)
    parser.add_argument('--verbose', default=False,  action="store_true")
    parser.add_argument('--image_only_model', default=False,  action="store_true")
    parser.add_argument('--no_training', default=False,  action="store_true")
    parser.add_argument('--run_prediction', default=False,  action="store_true")
    parser.add_argument('--source_dataset', default="GI", choices = ["GI","VG","OI","GCC","AIDASeedling"])
    parser.add_argument('--debug', default=False,  action="store_true")
    parser.add_argument('--exp_id', default=None, type=str)
    parser.add_argument('--final_act_layer', default= "softmax", choices = ["softmax" , "sigmoid"], type=str)
    parser.add_argument('--trainable_image_cnn', default= False, action ="store_true")
    parser.add_argument('--class_ct_threshold', default = 120, type = int)
    args = parser.parse_args()
    
    debug = args.debug
    verbose =args.verbose
    K.set_floatx('float32')


    #Depending on the source data copy the images to local storage  a subdir of /export/u10 
    dataset_localized = False
    minimal_train_set = False
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
    else:
        minimal_train_set = True
    # Form the experiment,training identifier
    train_file_id =os.path.basename(args.train_csv_file)
    train_file_id = os.path.splitext(train_file_id)[0]
    if args.image_only_model:
        train_file_id +='_image_only'
    if args.dataaug:
        train_file_id +='_aug'
    train_file_id +='_epoch_{}'.format(args.epoch)
    if args.exp_id is not None:
        output_id = args.exp_id+'_'+train_file_id
    else:
        output_id = train_file_id
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

    #REad the dataafreme which includes filepaths , captions and classnames
    train_df = pd.read_csv(args.train_csv_file, encoding='utf8')
    if verbose:
        print( train_df.apply(lambda x: pd.lib.infer_dtype(x.values)))
    
    
   

    texts = train_df["image_captions"].values.tolist()
    class_names_pd =  pd.unique(train_df["class"].values)
    init_classnames =class_names_pd.tolist()
    
    class_counts = train_df["class"].value_counts()
    class_counts.to_csv("{}_class_counts_orig.csv".format(args.source_dataset))
        
    class_ct_threshold = args.class_ct_threshold
    if minimal_train_set:
        class_ct_threshold = 0
    
    # limit_to_evaluable_classes = True
    # if limit_to_evaluable_classes:
    #     pd.read_tsv(
    #         "/nfs/mercury-11/u113/projects/AIDA/GoogleImageDownload_Rus_Scenario/all_image_concepts_GI_specific_translation_en_es_ru_uk_limit.csv",
    #         header =True)
    #     pd["Generic_annotation_label"]=
    #     pd[""]
    #     class_ct_threshold = 50

    #REmove any classes that have less # of examples than class_ct_threshold
    untrainable_classes    = class_counts < class_ct_threshold 
    untrainable_classnames = untrainable_classes[untrainable_classes].index.tolist()
    if verbose:
        print(untrainable_classnames)
        print (len(train_df))
    train_df = train_df.loc[~train_df['class'].isin(untrainable_classnames),:]
    
    labels_df= pd.read_csv("/nfs/mercury-11/u113/projects/AIDA/GI_Training_BBN.tsv",encoding='utf8',sep="\t")
    print(labels_df.columns)
    labels_df.columns  = ["row_index" , "class", "filenames", "page_title", "image_captions"]
    
    train_df.append(labels_df[["class", "filenames", "image_captions"]])

    #Update the filepaths if images were copied to local storage
    if dataset_localized :
        train_df =train_df.replace(KERAS_DATAGEN_DIR, LOCAL_STORAGE_DIR, regex= True)
    print (train_df["filenames"].head())
    KERAS_DATAGEN_DIR_EXTRA = "/nfs/raid66/u12/users/rbock/aida/image_captions/web_collection/20190110/aggregated"
    regex_exp = r'/nfs/raid66/u12/users/rbock/aida/image_captions/web_collection/20190110/aggregated(.*)'
    LOCAL_STORAGE_DIR = "/export/u10/sadali/AIDA/images/BBN_GI_minimal/image_data"
    replace_regex_exp = r'/export/u10/sadali/AIDA/images/BBN_GI_minimal/image_data\1'
    dataset_localized_extra = False
    if not os.path.exists (osp(LOCAL_STORAGE_DIR,"raw")):
        print ("copyying BBN-GI data from ",KERAS_DATAGEN_DIR_EXTRA, LOCAL_STORAGE_DIR)
        try:
            copytree(KERAS_DATAGEN_DIR_EXTRA, LOCAL_STORAGE_DIR)
            dataset_localized_extra = True
        except Exception as e:
            print (e)
            print ("Unable to copy image files for BBN-GI extra " )
            dataset_localized_extra = False
            print ("Removing any localized dirs")
            try:
                rmtree(osp(LOCAL_STORAGE_DIR,"raw"))
            except Exception as e:
                print (e)
        if dataset_localized_extra:
            train_df =train_df.replace(KERAS_DATAGEN_DIR_EXTRA, LOCAL_STORAGE_DIR, regex= True)


    print ("new examplar count {}".format(len(train_df)))
    classnames= [k for k in init_classnames if k not in untrainable_classnames] 
    if verbose:
        print("Num of classes ")
        print (len(classnames))
        print("Dimensions of training set dataframe")
        print(train_df.shape)
    new_class_counts = train_df["class"].value_counts()
    new_class_counts.to_csv("{}_class_counts.csv".format(args.source_dataset))
    with open("./models_dir/{}_classnames.json".format(train_file_id),"w") as json_fh:
        json.dump(dict(zip(classnames,range(len(classnames)))),json_fh)
    

    # Given image captions read from csv , compile the vocab(list of tokens)  for encoding the captions
    texts_ascii = [k.encode('ascii','ignore').decode() for k in texts]
    tokenizer = Tokenizer(num_words=args.maxtokencount)
    tokenizer.fit_on_texts(texts_ascii)
    
    print (type(texts[0]))
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    with open('./{}/keras_captiontokenizer_{}.pkl'.format(output_id,train_file_id),"wb")  as kfh:
        pkl.dump(tokenizer, kfh, protocol=pkl.HIGHEST_PROTOCOL)
    
    # Define  the whole model
    end2endmodel, vocab_map = \
            concept_detector( args.model_file, args.glove_embed_file,
                                        input_length=args.length, data_vocab = word_index,
                                        token_count = len(word_index),
                                        num_classes= len(classnames),
                                        image_only_model =args.image_only_model ,
                                        final_act = args.final_act_layer,
                                        trainable_early_layers= args.trainable_image_cnn)
    #optim_algo=Nadam(lr=.004 ,clipnorm=1.)
    optim_algo=Nadam(lr=.004 )
    #end2endmodel.compile(optimizer=optim_algo, loss="categorical_crossentropy")
    end2endmodel.compile(optimizer=optim_algo, loss="binary_crossentropy")

    # For debugging, print some weights
    with open("./{}/{}_trained_layer_weights_{}.txt".format(output_id,train_file_id,timestamp),"w") as fh:
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
            with open("./{}/caption_vocab-{}_{}.txt".format(output_id,args.source_dataset,timestamp),"w") as v_fh:
                for word in vocab_map:
                    v_fh.write("{}\n".format(word))
    

    if args.no_training:
        print ("Stopping before training")
        sys.exit(0)

    # Define the Keras data generator , based on the model input and whether data aug will be used
    train_datagen = None
    train_batch_size = 32 
    if args.image_only_model:
        if args.dataaug:
            train_datagen = IDG(width_shift_range = 0.2,zoom_range=0.2, height_shift_range=0.3 )
        else:
            train_datagen = IDG()
    else:
        train_batch_size = 24
        if args.dataaug:
            train_datagen = datagen(width_shift_range = 0.2,zoom_range=0.2, height_shift_range=0.3, rounds = 3 )
            train_batch_size = 24
        else:
            train_datagen = datagen()
    vis_sample_size = 200
    sampled_rows = train_df.sample(vis_sample_size)
    vis_input =  np.zeros (shape=(vis_sample_size,256,256),dtype="float32")
    vis_samples_classes= [""]*vis_sample_size
    if  args.image_only_model:
        if args.source_dataset == "GI":
            imagedir_root = LOCAL_STORAGE_DIR
        else:
            imagedir_root = LOCAL_STORAGE_DIR
        print (os.listdir(imagedir_root))
        print ("train_df columns")
        print (  train_df.describe())
        img_arrays         = map (lambda x:np.expand_dims(img_to_array(load_img(x, target_size=(256, 256))), axis=0),sampled_rows["filenames"].values.tolist())
        #cap_encoded_arrays = map (lambda x:np.expand_dims(tokenizer.texts_to_matrix(x), axis=0), sampled_rows["image_captions"].values.tolist())
        vis_input = np.concatenate(list(img_arrays),axis=0)
        #cap_input = np.concatenate(cap_encoded_arrays,axis=0)

        vis_input_list = [vis_input]
        
        
        train_data_it = train_datagen.flow_from_dataframe( 
                                                        dataframe= train_df,
                                                        directory= None,
                                                        x_col="filenames", y_col="class", has_ext=True,
                                                        target_size=(256, 256), color_mode='rgb',
                                                        classes=None, class_mode='categorical',
                                                        batch_size=train_batch_size, shuffle=False, seed=None,
                                                        save_to_dir=None,
                                                        save_prefix='',
                                                        save_format='png',
                                                        subset=None,
                                                        interpolation='nearest'
                                                        #,
                        #                 sort=False,
                    #                  follow_links= True
                                                        )
    else:
        cap_input =  np.zeros (shape=(vis_sample_size,len(tokenizer.word_index)))
        ro_i = 0
        print ("type(sampled_rows)",type(sampled_rows))
        img_arrays         = map (lambda x:np.expand_dims(img_to_array(load_img(x, target_size=(256, 256))), axis=0),sampled_rows["filenames"].values.tolist())
        cap_input = tokenizer.texts_to_matrix(sampled_rows["image_captions"].values.tolist())
        #cap_encoded_arrays = map (lambda x:np.expand_dims(tokenizer.texts_to_matrix(x), axis=0), )
        print (type(img_arrays))
        vis_input = np.concatenate(list(img_arrays),axis=0)
        #cap_input = np.concatenate(list(cap_encoded_arrays),axis=0)
        print (vis_input.shape)
        print (cap_input.shape)
        vis_input_list = [vis_input,cap_input]
        train_data_it = train_datagen.flow_from_dataframe( 
                                                        dataframe= train_df,
                                                        directory= None,
                                                        x_col=["filenames","image_captions"], 
                                                        y_col="class", has_ext=True,
                                                        target_size=(256, 256), color_mode='rgb',
                                                        classes=classnames, class_mode='categorical',
                                                        batch_size=train_batch_size, shuffle=False, seed=None,
                                                        save_to_dir=None,
                                                        save_prefix='',
                                                        save_format='png',
                                                        subset=None,
                                                        interpolation='nearest',
                                                        sort=False,
                                                        cap_token_vocab=word_index,
                                                        num_tokens = len(word_index),
                                                        follow_links= True)


    # Run the actual training  
    model_ckpt = ModelCheckpoint(filepath = './models_dir/{}/{}_ckpt_model-weights.{{epoch:02d}}'.format(output_id,train_file_id),monitor= "loss")
    tensorboard_logs_dir = "./models_dir/{}/tb_logs_{}".format(output_id,train_file_id)
    if not ose(tensorboard_logs_dir):
        os.makedirs(tensorboard_logs_dir)
    tb_callback = TensorBoard(log_dir = tensorboard_logs_dir,
                              embeddings_layer_names = ["l2_normalize_1","l2_normalize_2"],
                              embeddings_data=vis_input_list)
    callbacks_list = [model_ckpt,tb_callback]

    # save the model config under models_dir as json
    if not os.path.exists("models_dir/{}".format(output_id)):
        os.makedirs("models_dir/{}".format(output_id))
    model_fname = "{}_keras_vse_model-{}".format(train_file_id,timestamp)
    try:
        with open("./models_dir/{}/{}.json".format(output_id,model_fname),"w") as json_fh:
            json_fh.write(end2endmodel.to_json()+"\n")
    except Exception as e:
        print (e)
        print ("Unable to save model as json files")
    class_indices_for_model = train_data_it.class_indices
    with open ("./models_dir/{}/{}_{}_class_indices.json".format(output_id, model_fname,class_ct_threshold),"w") as json_fh:
        json.dump(dict(train_data_it.class_indices),json_fh)

    if debug:
        end2endmodel.fit_generator(train_data_it,steps_per_epoch=200)
    else:
        end2endmodel.fit_generator(train_data_it,callbacks=callbacks_list, epochs=args.epoch)
    
    
    # save the model under models_dir
    end2endmodel.save("./models_dir/{}/{}.h5".format(output_id,model_fname))
    try:
        with open("./models_dir/{}/{}.json".format(output_id , model_fname),"w") as json_fh:
            json_fh.write(end2endmodel.to_json()+"\n")
        end2endmodel.save_weights("./models_dir/{}/{}_weights.h5".format(output_id, model_fname))
    except Exception as e:
        print (e)
        print ("Unable to save model as json+h5 files")


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


    test_preds_on_train_set = args.run_prediction
    
    if test_preds_on_train_set:
        # Actually run the prediction on the training test.
        predictions = end2endmodel.predict_generator(test_data_it)
        preds_out = open("./{}/{}_{}preds_out.txt".format(output_id,train_file_id,timestamp),"w")
        for pr in predictions:
            print(pr)
            preds_out.write("{}\n".format(pr))
