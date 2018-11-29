from models import *
from keras.layers import Concatenate,Dense
import pandas






def concept_detector(args):

    image_encoder, sentence_encoder, vocab_map , image_feat_extractor = \
        build_pretrained_models(args.model_file, input_length=args.length)
    
    captioned_image_descriptor  = Concatenate(image_encoder,sentence_encoder)

    concept_detector_scores = Dense(num_classes )(captioned_image_descriptor)

    end_to_end_model = Model(inputs= [image_feat_extractor.inputs ,sentence_encoder.inputs], outputs = concept_detector)
    return end_to_end_model



if __name__=="__main__":

    parser = argparse.ArgumentParser('Visual semantic embeddings')
    parser.add_argument('--model_file', type=six.text_type, required = True)
    parser.add_argument('--length', type=int, default=None)
    parser.add_argument('--train_model_file', type=int, required=True)
    args = parser.parse_args()

    trainable_model = concept_detector(args)
    trainable_model.compile(optimizer='adam', loss='binary_crossentropy')
    
    
