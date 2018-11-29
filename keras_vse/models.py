import numpy as np
import six
from keras import backend as K
from keras.layers import Convolution2D, Dense, Embedding, GRU, Input,Concatenate
from keras.models import Model, Sequential
from keras.applications.vgg19 import VGG19
from layers import L2Normalize
from tools import encode_sentences

import pandas


def concept_detector(model_file, input_length ,data_vocab ,token_count):

    image_encoder, sentence_encoder, vocab_map , image_feat_extractor = \
        build_pretrained_models(model_file, input_length,data_vocab = data_vocab,token_count=token_count)
    captioned_image_descriptor  = Concatenate(image_encoder,sentence_encoder)

    concept_detector_scores = Dense(num_classes )(captioned_image_descriptor)
    for enc in image_encoder, sentence_encoder:
        enc.compile(optimizer='nadam', loss='mse')
    end_to_end_model = Model(inputs= [image_feat_extractor.inputs ,sentence_encoder.inputs], outputs = concept_detector)
    return end_to_end_model,vocab_map



def build_image_feat_extractor():

    return  VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling=None)

def build_image_encoder(weights=None, input_dim=4096, embedding_dim=1024, normalize=True):
    input = Input(shape=(input_dim,))
    x = Dense(
        embedding_dim,
        weights=weights
    )(input)
    if normalize:
        x = L2Normalize()(x)
    model = Model(input=input, output=x)
    return model


def build_sentence_encoder(embedding_weights=None, gru_weights=None, input_length=None, vocab_dim=32198,
        vocab_embedding_dim=vocab_embed_dim, embedding_dim=1024, normalize=True, finetune_word_embedding=False):
    # NOTE: This gives slightly different results than the original model.
    # I think it's because the original has a different masking scheme.
    model = Sequential([
        Embedding(
            vocab_dim, vocab_embedding_dim, input_length=input_length,
            weights=embedding_weights, mask_zero=True,   # TODO: masking isn't quite right,
            trainable = finetune_word_embedding
        ),
        GRU(embedding_dim, weights=gru_weights, inner_activation='sigmoid'),
    ])
    if normalize:
        model.add(L2Normalize())
    return model


def build_pretrained_models(model_filename, input_length=None,data_vocab = None, token_count=None, normalize=True):
    if model_filename is not None:
        img_enc_weights, embedding_weights, gru_weights, init_vocab_map = load_pretrained_parameters(model_filename)
    glove_embedding_mat = None
    if token_count is None:
        print("assuming original dict pkl is available for pretrained model")
        token_count = len(init_vocab_map)
        glove_embedding_mat= embedding_weights
    glove_embedding_mat = compute_embedding_matrix(glove_file,data_vocab,None)
    vocab_embed_dim = glove_embedding_mat.shape[1]
    image_feat_extractor = build_image_feat_extractor()
    image_encoder = build_image_encoder(weights=img_enc_weights, normalize=normalize)
    
    sentence_encoder = build_sentence_encoder(
        embedding_weights=glove_embedding_mat,
        gru_weights=gru_weights,
        input_length=input_length, vocab_dim=token_count,
        vocab_embedding_dim=vocab_embed_dim,
        normalize=normalize)
    return image_encoder, sentence_encoder, data_vocab , image_feat_extractor


def load_pretrained_parameters(filename):
    '''Load up the pre-trained weights from the @ryankiros implementation.
    '''
    params = np.load(filename)
    vocab_map = np.load('{}.dictionary.pkl'.format(filename))
    # image encoder weights
    if params:
        img_enc_weights = [params['ff_image_W'], params['ff_image_b']]
    else:
        img_enc_weights = None
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
    return img_enc_weights, embedding_weights, gru_weights, vocab_map

def compute_embedding_matrix(glove_file,word_index,vocab_embed_dim=None):
    embeddings_index = dict()
    f= open(glove_file,'r')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        vocab_embed_dim = len(values[1:])
        embeddings_index[word] = coefs
    f.close()
    embedding_matrix = np.zeros((len(word_index) + 1, vocab_embed_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            
            print(word + " not found in glove embedding. translate word to english if foreign word")
    return embedding_matrix

