import os 
from  os.path import join  as osp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def caption_image(image_file_path,concept_score_triples,output_dir,caption_threshold = 0.1 ,trans_dict=None):
    '''
    caption image with detected concepts and scores
    '''
    try:
        img = mpimg.imread(image_file_path)
    except Exception as e:
        print (str(image_file_path) + " is not readable as image")
        return
    fileprefix = '.'.join(os.path.basename(image_file_path).split('.')[:-1])
    plt.axis('off')
    plt.imshow(img)
    capt = ''
    max_det = 5
    det = 0
    for triple in concept_score_triples:
        if float(triple[2]) < caption_threshold:
            continue
        det += 1
        if trans_dict is not None:
            if triple[0] in trans_dict.keys():
                triple = (trans_dict[triple[0]],str(triple[1]),"{:02f}".format(triple[2]))
            else:
                print ("Warning: concept id is not among  the ids listed in the translation file")

        triple_p = [str(triple[0]),str("{:02f}".format(triple[2]))]
        capt += " | "+' '.join(triple_p)
        if det  >= max_det:
            break
    plt.title(capt)
    plt.savefig(osp(output_dir,fileprefix+'.jpg'),bbox_inches="tight")
