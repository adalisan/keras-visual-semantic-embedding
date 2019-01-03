#!/usr/bin/env python3
import argparse
import codecs
import csv
import json
import os
from os.path import exists as ose
from os.path import join as osp

import numpy as np
import pandas as pd
import six
import visual_genome
from gensim.models import Word2Vec
from glove import Glove
from PIL import Image
from visual_genome import api as vg
from visual_genome import local as vg_local

from vocab import *

try:
    from urllib.parse import urlparse as urlparse
except  ImportError:
    import urlparse
    

import glob
import xml.etree.ElementTree as exml
from os.path import exists as ose

image_path_dict ={}
image_dict = {}
image_caption_dict ={}
image_concept_dict={}
concept_image_dict = {}
def GI_download_verify_images(fpath_dict_json,img_caption_dict_json,img_class_json, GI_staging_dir):
    
    with open(osp(GI_staging_dir,fpath_dict_json),'r') as fp:
        img_paths_to_hash=json.load(fp)
    with open(osp(GI_staging_dir,img_caption_dict_json),'r') as fp:
        img_captions_dict=json.load(fp)
    with open(osp(GI_staging_dir,img_class_json),'r') as fp:
        img_concepts_dict=json.load(fp)
    with open(osp(GI_staging_dir,img_class_to_img_fpath_json),'r') as fp:
        img_concept_examplar_dict=json.load(fp)

    delete_from_dataset = []
    img_paths_to_hash = dict((k,v) for k,v in six.iteritems(img_paths_to_hash) if v not in delete_from_dataset)
    img_captions_dict = dict( (k,v) for k,v in six.iteritems(img_captions_dict) if k not in delete_from_dataset)
    img_concepts_dict = dict( (k,v) for k,v in six.iteritems(img_concepts_dict) if k not in delete_from_dataset)
    img_concept_examplar_dict = dict( (k,v) for k,v in six.iteritems(img_concepts_dict) if k not in delete_from_dataset)


    with codecs.open("GI_Rus_Ven_concept_img_lut.json",'w',encoding='utf8') as fp:
        json.dump(file_concept_dict, fp,ensure_ascii=False)
    with codecs.open("GI_Rus_Ven_img_hash_img_fpath_lut.json",'w',encoding='utf8') as fp:
        json.dump(img_paths_to_hash, fp,ensure_ascii=False)
    with codecs.open("GI_Rus_Ven_img_hash_concept_lut.json",'w',encoding='utf8') as fp:
        json.dump(class_lut, fp,ensure_ascii=False)
    with codecs.open("GI_Rus_Ven_img_miss_img.json",'w',encoding='utf8') as fp:
        json.dump(unk_img,fp,ensure_ascii=False)
    with codecs.open("GI_Rus_Ven_img_desc_cap_lut.json","w",encoding='utf8') as fp:
        json.dump(image_descriptions,fp,ensure_ascii=False)


def GI_download_ingest(KERAS_DATAGEN_DIR,
                        GI_staging_dir = "/nfs/mercury-11/u113/projects/AIDA/GoogleImageDownload_Rus_Scenario",
                        fpath_dict_json= "GI_Rus_Ven_img_hash_img_fpath_lut.json",
                        img_caption_dict_json="GI_Rus_Ven_img_desc_cap_lut.json",
                        img_class_json= "GI_Rus_Ven_img_hash_concept_lut.json",
                        img_class_to_img_fpath_json= "GI_Rus_Ven_concept_img_lut.json"
                        ):
    img_paths ={}
    with open(osp(GI_staging_dir,fpath_dict_json),'r') as fp:
        img_paths_to_hash=json.load(fp)
    with open(osp(GI_staging_dir,img_caption_dict_json),'r') as fp:
        img_captions_dict=json.load(fp)
    with open(osp(GI_staging_dir,img_class_json),'r') as fp:
        img_concepts_dict=json.load(fp)
    with open(osp(GI_staging_dir,img_class_to_img_fpath_json),'r') as fp:
        img_concept_examplar_dict=json.load(fp)
    keras_train_df = pd.DataFrame(data=dict({"filenames":np.array([],dtype="<U2"),
                                    "image_captions":np.array([],dtype="<U2"),
                                    "class":np.array([],dtype="<U2")}))
                                    # keras_train_df = pd.DataFrame(data=dict({"filenames":np.array([],shape=(0,),dtype="<U2"),
                                    # "image_captions":np.array([],shape=(0,),dtype="<U2"),
                                    # "class":np.array([],shape=(0,),dtype="<U2")}))
    invalid_symlinks = []
    for concept,img_paths in img_concept_examplar_dict.items():
        concept= concept.strip("'").strip('"')
        concept_ex_dir = osp(KERAS_DATAGEN_DIR,concept)
        if not ose(concept_ex_dir):
            os.makedirs(concept_ex_dir)
        img_symlink_paths = []
        for img_path in img_paths:
            img_ext= os.path.splitext(img_path)[1]
            img_hash= img_paths_to_hash[img_path]
            img_hash_fname = img_hash+img_ext
            img_symlink_path= osp(concept_ex_dir,img_hash_fname)
            if not ose(img_symlink_path):
                try:
                    os.symlink(src=img_path, dst=img_symlink_path)
                except Exception as e:
                    print(e)
                    invalid_symlinks.append(img_symlink_path)

            img_symlink_paths.append(img_symlink_path)
            
        new_df_dict = {"filenames":img_symlink_paths,
        
        "image_captions": [img_captions_dict[img_paths_to_hash[img_path]] for img_path in img_paths  ] ,
        "class" : [concept.strip('\"').strip("'") for i in img_paths]}
        keras_train_df = keras_train_df.append(pd.DataFrame(new_df_dict))
    print(keras_train_df.tail())
    keras_train_df.to_csv("/nfs/mercury-11/u113/projects/AIDA/GI_keras_train.csv",encoding='utf8')



def visual_genome_ingest(data_dir = "/nfs/mercury-11/u113/projects/AIDA/VisualGenomeData",
                        image_data_dir =  "/nfs/mercury-11/u113/projects/AIDA/VisualGenomeData/image_data" ):
    # with open(osp(data_dir,"relationships.json"),'r') as fp:
    #     relations=json.load(fp)
    # with open(osp(data_dir,"relationship_synsets.json"),'r') as fp:
    #     relation_synsets=json.load(fp)

    # with open(osp(data_dir,"objects.json"),'r') as fp:
    #     objects = json.load(fp)
    # with open(osp(data_dir,"objects_synsets.json"),'r') as fp:
    #     object_synsets = json.load(fp)
    # with open(osp(data_dir,"region_descriptions.json"),'r') as fp:
    #     region_descriptions = json.load(fp) 
    # with open(osp(data_dir,"image_data.json"),'r') as fp:
    #     image_data = json.load(fp)
    # for img_obj_pairs in  objects:
    #     img_id  = img_obj_pairs["image_id"]
    #     objects = img_obj_pairs["objects"]
    #     for obj in objects:
    #         class_names = obj["synsets"]

    keras_train_df = pd.DataFrame(data=dict({"filenames":np.array([],dtype="<U2"),
                                    "image_captions":np.array([],dtype="<U2"),
                                    "class":np.array([],dtype="<U2")}))
    print("saving scen graphs")
    #vg_local.save_scene_graphs_by_id(data_dir=data_dir+'/' ,image_data_dir='{}/by-id/'.format(data_dir))
    print("loading scene graphs by image id")
  # Load scene graphs in 'data/by-id', from index 0 to 200.
  # We'll only keep scene graphs with at least 1 relationship.
    #image_ids = vg.get_all_image_ids()
    print("got all image ids")
    #all_image_data = vg_local.get_all_image_data(data_dir=data_dir)
    image_data_file = osp(data_dir,'image_data.json')
    image_data_dict = json.load(open(image_data_file))
    all_image_data_dict =  dict((k['image_id'], k['url']) for k in image_data_dict )
    image_ids = all_image_data_dict.keys()
    print ("There are {} images in VG dataset".format(len(image_ids)))
    for img_it,img_id in enumerate(image_ids):
        scene_graphs = vg_local.get_scene_graphs(start_index=img_id, end_index=img_id+1, min_rels=1,max_rels= 5,
                                    data_dir=data_dir+'/', image_data_dir='{}/by-id/'.format(data_dir))
        if img_it % 10000 == 0 :
            print("{}th image and captions added to dataframe".format(img_it))
        if len(scene_graphs)==0:
            continue
        scene_graph = scene_graphs[0]
        #scene_graph = get_scene_graph(img_id)
        
        rels_img =scene_graph.relationships
        objects = scene_graph.objects
        
        if (isinstance(rels_img,list)):
            print (len(rels_img))
        # for rels in rels_img:
        #     print ("type of rels")
        #     print (type(rels))
        
        img_captions = ""
        for rels in rels_img:
            rel_num,rel_str = str(rels).split(':')

            img_captions += rel_str +' '
        
        img_classes =  [str(obj) for obj in scene_graph.objects]
        num_of_multilabels = len(img_classes)
        
        
        z = urlparse( all_image_data_dict[img_id])
        _,subdir,filename = z.path.rsplit('/',maxsplit=2)
        img_path = osp(image_data_dir,subdir,filename)
        used_img_captions = [img_captions  for idx in range(num_of_multilabels)]
        

        new_df_dict = {"filenames":[img_path for i in range(num_of_multilabels)],
        "image_captions": used_img_captions,
        "class" : img_classes}
        keras_train_df = keras_train_df.append(pd.DataFrame(new_df_dict))
    keras_train_df.to_csv("/nfs/mercury-11/u113/projects/AIDA/VG_keras_train.csv",encoding="utf8",index=False)


def get_similar(model,tok,topn):
    if '_' in tok:
        tok = tok.replace('_',' ')
    # if '-' in tok:
    #     tok = tok.replace('-',' ')
    if tok not in model:
        return []
    else:
        return model.most_similar(tok,topn=topn)

def get_img_concepts_OI(  caption_vocab , class_labels_csv = "../../Corpora_and_Concepts/Combined_OpenCorpora_new_OI_Sing_VW_train_labels.csv",
                         img_id_imgpath_csv= "/nfs/mercury-11/u113/projects/AIDA/Comb_YT8m_Sing_newOI.part" , sim_thres =0.6,
                         glove_model_file = "/nfs/mercury-11/u113/projects/AIDA/glove.840B.300d.txt"):

    train_df = pd.DataFrame(data=dict({"filenames":np.array([],dtype="<U2"),
                                    "image_captions":np.array([],dtype="<U2"),
                                    "class":np.array([],dtype="<U2")}))

    import gensim.downloader as api
    
    try:
        model_download_path = api.load('glove-twitter-100',return_path=True)
        print("using model file at  ", model_download_path)
    except Exception as e:
        print (e)
    word2vec_model = api.load('glove-twitter-100',return_path=False)
    
    print(type(word2vec_model))
    #word2vec_model.most_similar
    #glove_model = Glove.load(glove_model_file)
    #Word2Vec.load()
    topn = 3
    img_id_imgpath_dict = dict()
    with open(img_id_imgpath_csv,'r') as fh:
        for line in fh.readlines():
            line = line.strip()
            img_id,img_path = line.split(':')
            img_id_imgpath_dict.update({img_id: img_path})
    img_id_classname_dict =dict()
    with open(class_labels_csv,'r') as fh:
        for line in fh.readlines():
            line = line.strip()
            img_id,classes_str =line.split(',')
            classes_str= classes_str.strip()
            classnames = classes_str.split(' ')
            img_id_classname_dict[img_id] = classnames
            tokens = []
            
            src_tokens_pairs  = [cl.split('-') for cl in classnames ]
            for pair in src_tokens_pairs:
                if len(pair)>1:
                    tokens.append(pair[1])
                else:
                    print("weird classname")
                    print(pair)

            #glove.most_similar('token', )
            dummy_caption = ""
            
            similar_tokens_list = [get_similar(word2vec_model,tok,topn=topn) for tok in tokens ]
            token_ct = 0
            for similar_tokens in similar_tokens_list:
                for similar_token,similarity in similar_tokens:

                    if similarity > sim_thres and similar_token in caption_vocab:
                        dummy_caption += " {}".format(similar_token)
                        token_ct += 1
            
            img_path = img_id_imgpath_dict.get(img_id,"")
            new_df_dict = { "filenames":      [img_path for i in classnames] ,
                            "image_captions": [dummy_caption for i in classnames]  ,
                            "class" :         [cl for cl in classnames]
                        }
            train_df = train_df.append(pd.DataFrame(new_df_dict))
    train_df.to_csv("/nfs/mercury-11/u113/projects/AIDA/OI_keras_train.csv",encoding="utf8")
            



def BBN_AIDA_annotation_ingest(
        dataset_dir = "/nfs/raid66/u12/users/rbock/aida/image_captions/annotation_of_seedling_corpus/spreadsheets/with_paths" ,
        annotation_dir = "/nfs/raid66/u12/users/rbock/aida/image_captions/annotation_of_seedling_corpus/images/"
        ):
    col_names = ["child_id", "article_url","image_url","image_caption","filenames"]
    annot_sets = [ "LDC2018E01", "LDC2018E52" ]
    image_id_classlabel_dict = {}
    for dataset_file  in glob.glob(osp(dataset_dir,"*.tab")):

        f_id = os.path.basename(dataset_file)
        f_id = f_id.splitext()[0]
        for annot_file  in glob.glob(osp(annotation_dir,f_id,"*.xml")):
            tree=exml.parse(annot_file)
            root=tree.getroot()
            fname = root[1]
            image_id = fname.splitext()[0]
            
            classnames = [ obj.name.tostring()  for obj in root.findall('./object') ]
            image_id_classlabel_dict .update({image_id:classnames})

        label_df = pd.DataFrame( [ (k,v_el)  for k,v in image_id_classlabel_dict.items() for v_el in v ] ,
                    columns=["image_id","class"]) 
        trainset_df = pd.read_csv(dataset_file,dialect='excel-tab', header = None, names=col_names)
        merged_df= trainset_df.merge(label_df,left_on="child_id",right_on = "image_id",how = "left"  )

        merged_df.subset(["filenames","image_caption","class"]).to_csv("AIDA_seedling_keras_train_{}.csv".format(f_id),
            encoding="utf8")
    return merged_df




def Google_conceptual_captions_ingest(image_caption_tsv, kfile):
    with open(kfile, 'r') as fh:
        keywords  = [ z.strip() for z in fh.readlines() ]
    
    tokens_set = set()
    with open(out_tsv_file, 'w') as fh:
        with open(out_vocab_file, 'w') as fh:
            for row,new_tokens in  getrows(image_caption_tsv,keywords):
                out_tsv_file.write(row)
                tokens_set.update(new_tokens)


def getrows(filename, keywords):
    with open(filename, "rb") as csvfile:
        datareader = csv.reader(csvfile, delimiter ='\t' )
        yield next(datareader)  # yield the header row
        count = 0
        for row in datareader:
            tokens = row[0].strip().split(' ')
            for token in tokens:
                if token in keywords or token+'s' in keywords:
                    yield row,tokens
                    count += 1
    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Preprocess ')
    parser.add_argument('--out_data_dir', type=str,dest= "KERAS_DATAGEN_DIR")
    parser.add_argument('--dataset', type=str,dest= "data_source", choices= ["GI","GoogleCaps","VisualGenome","OI"])

    args = parser.parse_args()
    if args.data_source=="GI":
        GI_download_ingest(args.KERAS_DATAGEN_DIR)
    elif args.data_source == "GoogleCaps":
        Google_conceptual_captions_ingest(image_caption_tsv= "/nfs/mercury-11/u113/projects/AIDA/Train%2FGCC-training.tsv",
                                    kfile="/nfs/mercury-11/u113/projects/AIDA/concept_lists/relevant_vocab.txt.3")
    elif args.data_source == "OI":
        caption_vocab_file = "caption_vocab.txt"
        caption_vocab = []
        with open(caption_vocab_file,"r") as cap_v_file:
            for word in  cap_v_file.readlines():
                caption_vocab.append(word.strip())
        get_img_concepts_OI(caption_vocab=caption_vocab)
    else:
        visual_genome_ingest()



    # with open(args.caption_json_file, "r") as caption_file_fh:
    #     caption_dict = json.load(caption_file_fh)
    #     build_dictionary(caption_dict.items())
