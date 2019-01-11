#encoding: utf-8
import numpy as np
import six
from keras import backend as K
from keras.layers import Convolution2D, Dense, Embedding, GRU, Input,Concatenate, Dropout
from keras.models import Model, Sequential
from keras.applications.vgg19 import VGG19
from layers import L2Normalize
from keras.utils import plot_model
from keras import __version__ as keras_ver
from tools import encode_sentences

import pandas
#from glove import Glove

# from translate import Translator
import pdb
try:
    import cPickle as pickle
except ImportError:
    import pickle

import h5py

# def ModelwithMetadata(Model):
#     def __init__(self,metadata_json=None):
#         super(ModelwithMetadata,self).__init__(name='KerasModelwithMetadata'))
#     def save()
#         ModelwithMetadata

def concept_detector(model_file, glove_file, input_length ,data_vocab ,
                     token_count, num_classes, 
                     image_only_model =False,dropout_before_final = 0.0,
                     final_act="sigmoid",
                     trainable_early_layers=False):

        
        #, embedding_weights, gru_weights, init_vocab_map = 
    if image_only_model:
        vocab_map = None
        img_enc_weights = None
        if model_file is not None:
            img_enc_weights = load_pretrained_parameters(model_file)
        image_encoder,image_feat_extractor = build_image_encoder(weights=img_enc_weights,  
                                                                 embedding_dim=1024, normalize=True,
                                                                 trainable_early_layers=trainable_early_layers)
        if dropout_before_final> 0.0 :
            image_encoder_out = Dropout(dropout_before_final) (image_encoder.outputs[0])
        else:
            image_encoder_out = image_encoder.outputs[0]
        concept_detector_scores = Dense(num_classes, activation=final_act )(image_encoder_out)
        image_encoder.compile(optimizer='nadam', loss='mse')
        end_to_end_model = Model(inputs= [image_feat_extractor.inputs[0] ],
                             outputs = concept_detector_scores)
        plot_model(end_to_end_model, to_file='model.png')
        return end_to_end_model,vocab_map
    
    image_encoder, sentence_encoder, vocab_map , image_feat_extractor = \
        build_pretrained_models(model_file, glove_file,input_length,
                                data_vocab = data_vocab,token_count=token_count,
                                trainable_early_layers=False)
    captioned_image_descriptor  = Concatenate(axis=-1)([image_encoder.outputs[0],
                                                        sentence_encoder.outputs[0]])
    if dropout_before_final > 0.0 :
        captioned_image_descriptor = Dropout(dropout_before_final) (captioned_image_descriptor)

    concept_detector_scores = Dense(num_classes , activation=final_act)(captioned_image_descriptor)
    for enc in image_encoder, sentence_encoder:
        enc.compile(optimizer='nadam', loss='mse')
    end_to_end_model = Model(inputs= [image_feat_extractor.inputs[0] ,
                                      sentence_encoder.inputs[0]],
                             outputs = concept_detector_scores)
    plot_model(end_to_end_model, to_file='model.png')
    return end_to_end_model,vocab_map



def build_image_feat_extractor():
    return  VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling="avg")

def build_image_encoder(weights=None, input_dim=4096, embedding_dim=1024, normalize=True, trainable_early_layers=False ):
    #input  = Input(shape=(input_dim,))
    init_model = build_image_feat_extractor()
    for layer in init_model.layers:
        layer.trainable = trainable_early_layers
    #init_model.compile
    x= Dense(
        embedding_dim,
        weights=weights
    )(init_model.outputs[0])
    if normalize:
        x = L2Normalize()(x)
    model = Model(input=init_model.inputs[0], output=x)
    
    plot_model(model, to_file='image_model.png')
    return model, init_model

def build_image_encoder_seq_api(weights=None, input_dim=4096, embedding_dim=1024, normalize=True):
    model = build_image_feat_extractor()
    
    model.add(Dense(
        embedding_dim,
        weights=weights
    ))
    if normalize:
        model.add(L2Normalize())
    
    return model


def build_sentence_encoder(embedding_weights=None, gru_weights=None, input_length=None, vocab_dim=32198,
        vocab_embedding_dim=300, embedding_dim=1024, normalize=True, finetune_word_embedding=False):
    # NOTE: This gives slightly different results than the original model.
    # I think it's because the original has a different masking scheme.
    model = Sequential([
        Embedding(
            input_dim=vocab_dim, output_dim= vocab_embedding_dim, input_length=input_length,
            weights=embedding_weights, mask_zero=True,   # TODO: masking isn't quite right,
            trainable = finetune_word_embedding
        ),
        GRU(embedding_dim, weights=gru_weights, inner_activation='sigmoid'),
    ])
    if normalize:
        model.add(L2Normalize())
    return model


def build_pretrained_models(model_filename, glove_file,input_length=None,data_vocab = None,
                             token_count=None, normalize=True, trainable_early_layers =False):
    gru_weights= None
    img_enc_weights = None
    
    if model_filename is not None:
        img_enc_weights = load_pretrained_parameters(model_filename)
        #, embedding_weights, gru_weights, init_vocab_map = 
    
    glove_embedding_mat = None
    #reading weights from original vse(@ryankiros) trained model (coco,flickr8k or flickr30k)
    if token_count is None:
        print("assuming original dict pkl is available for pretrained model")
        embedding_weights, gru_weights, init_vocab_map =load_pretrained_embedding_weights(model_filename)
        token_count = len(init_vocab_map)
        glove_embedding_mat= embedding_weights
    glove_embedding_mat,_ = compute_embedding_matrix(glove_file,data_vocab,None)
    vocab_embed_dim = glove_embedding_mat.shape[1]
    

    image_encoder,image_feat_extractor = build_image_encoder(weights=img_enc_weights, 
                                                             embedding_dim=1024, normalize=normalize,
                                                             trainable_early_layers= trainable_early_layers)
    print ("Word Embedding matrix shape")
    print(glove_embedding_mat.shape)
    #print(glove_embedding_mat[10,:])
    sentence_encoder = build_sentence_encoder(
        embedding_weights=[glove_embedding_mat],
        gru_weights=gru_weights,
        input_length=input_length, vocab_dim=token_count+1,
        vocab_embedding_dim=vocab_embed_dim,
        normalize=normalize)
    return image_encoder, sentence_encoder, data_vocab , image_feat_extractor


def load_pretrained_parameters(filename):
    '''Load up the pre-trained weights from the @ryankiros implementation.
    '''
    params = np.load(filename)
    
    # image encoder weights
    if params:
        img_enc_weights = [params['ff_image_W'], params['ff_image_b']]
    else:
        img_enc_weights = None
    return img_enc_weights

def load_pretrained_embedding_weights(filename):
    params = np.load(filename)
    vocab_map = np.load('{}.dictionary.pkl'.format(filename))
    # sentence encoder weights
    embedding_weights = [params['Wemb']]
    W_h = params['encoder_Wx']
    U_h = params['encoder_Ux']
    b_h = params['encoder_bx']
    W_r, W_z = np.split(params['encoder_W'], 2, axis=1)
    U_r, U_z = np.split(params['encoder_U'], 2, axis=1)
    b_r, b_z = np.split(params['encoder_b'], 2)
    gru_weights = [
        W_z, U_z, b_z,
        W_r, U_r, b_r,
        W_h, U_h, b_h,
    ]
    return embedding_weights, gru_weights, vocab_map

def compute_embedding_matrix(glove_file,word_index,vocab_embed_dim=None):
    embeddings_index = dict()
    vocab_embed_dim = 0
    avg_embed_vec = None
    if glove_file.endswith('pkl') :
        num_embeds = 0
        avg_embed_vec = None
        embeddings_index=pickle.load(open(glove_file,'rb'))
        #print (embeddings_index.keys())
        for _,avec in embeddings_index.items():
            vocab_embed_dim  = avec.size
            if avg_embed_vec is None:
                avg_embed_vec = avec.copy()
            else:
                avg_embed_vec += avec 
            num_embeds +=1
        avg_embed_vec = avg_embed_vec/num_embeds
    elif glove_file.endswith('h5') :
        glove_hdf5 = h5py.File(glove_file,'r')
        embeddings_index = glove_hdf5["embed_dict"]
        avg_embed_vec = embeddings_index["avg_word_vector"].copy()
        vocab_embed_dim = avg_embed_vec.size
        del embeddings_index["avg_word_vector"]
        

    else:
        f= open(glove_file,'r')
        avg_embed_vec = None
        num_embeds = 0
        for line in f:
            values = line.split(' ')
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float32')
            except Exception as e:
                print (e)
                print (values[0])
                print (values[1])
                print (values[-1])
                print (line)
            vocab_embed_dim = len(values[1:])
            embeddings_index[word] = coefs
            if avg_embed_vec is None:
                avg_embed_vec = coefs.copy()
                
            else:
                avg_embed_vec += coefs 
            num_embeds +=1
        avg_embed_vec = avg_embed_vec/num_embeds

        f.close()
        f= open(glove_file+'.pkl','wb')
        pickle.dump(embeddings_index, f)
        f.close()
        with h5py.File(glove_file+'.h5','w') as f:
            embed_dict=f.create_group("embed_dict")
            for word, i in word_index.items():
                embed_dict[word] = coefs
            embed_dict["avg_word_vector"] = avg_embed_vec
    embedding_matrix = np.zeros((len(word_index) + 1, vocab_embed_dim))
    fh = open("oov.txt",'w', encoding="utf8")
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i,:] = embedding_vector
        else:
            #translator= Translator(to_lang="english")
            #translation = translator.translate(word)
            #pdb.set_trace()
            #print( u' %s not found in glove embedding. translating word to english in case foreign word' % word)
            
            fh.write("%s\n" % word)

            embedding_matrix[i,:] = avg_embed_vec

            #print (translation)
            #pdb.set_trace()
            # embedding_vector = embeddings_index.get(translation)
            # if embedding_vector is not None:
            #     embedding_matrix[i] = embedding_vector
    embedding_matrix[-1,:] =avg_embed_vec
    fh.close()
    return embedding_matrix,vocab_embed_dim

