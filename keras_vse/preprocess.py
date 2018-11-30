#!/usr/bin/env python3
from  vocab import *
import os
import json 
import csv
import argparse
from os.path import join as osp
import pandas as pd
from urllib.parse import urlparse as urlparse
from visual_genome import api as vg
from os.path import exists as ose
import numpy as np


image_path_dict ={}
image_dict = {}
image_caption_dict ={}
image_concept_dict={}
concept_image_dict = {}

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
                        image_data_dir =  "/nfs/mercury-11/u116/projects/AIDA/VisualGenomeData/image_data" ):
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

    keras_train_df = pds.DataFrame({"filenames":np.array(dtype="<U2"),
                                    "image_captions":np.array(dtype="<U2"),
                                    "class":np.array(dtype="<U2")})
    image_ids = vg.get_all_imageids()
    for img_id in image_ids:
        scenegraph = vg.get_scene_graph_of_image(id=img_id)
        regions = vg.get_region_descriptions_of_image(id=img_id)
        #select the first relationship at random
        rels_img =scenegraph.relationships
        img_captions =  rels_img.values()
        img_classes =  scenegraph.objects
        num_of_multilabels = len(img_classes)
        
        image = vg.GetImageData(id=image_id)
        z = urlparse( image.url)
        _,subdir,filename = z.path.rsplit('/',maxsplit=2)
        img_path = osp(image_data_dir,subdir,filename)
        if len(img_captions)>num_of_multilabels:
            used_img_captions = np.random.choice(img_captions,k=num_of_multilabels)
        else:
            used_img_captions = [img_captions [0]]

        new_df_dict = {"filenames":img_path,
        "image_captions": used_img_captions,
        "class" : img_classes}
        keras_train_df = keras_train_df.append(new_df_dict)
    keras_train_df.to_csv("/nfs/mercury-11/u113/projects/AIDA/VG_keras_train.csv",encoding="utf8")

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
    parser.add_argument('--dataset', type=str,dest= "data_source", choices= ["GI","GoogleCaps","VisualGenome"])

    args = parser.parse_args()
    if args.data_source=="GI":
        GI_download_ingest(args.KERAS_DATAGEN_DIR)
    elif args.data_source == "GoogleCaps":
        Google_conceptual_captions_ingest(image_caption_tsv= "/nfs/mercury-11/u113/projects/AIDA/Train%2FGCC-training.tsv",
                                    kfile="/nfs/mercury-11/u113/projects/AIDA/concept_lists/relevant_vocab.txt.3")
    else:
        visual_genome_ingest()



    # with open(args.caption_json_file, "r") as caption_file_fh:
    #     caption_dict = json.load(caption_file_fh)
    #     build_dictionary(caption_dict.items())
