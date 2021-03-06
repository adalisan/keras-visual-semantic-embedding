#encoding: utf-8


import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator as IDG
from keras.preprocessing.image import Iterator

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img

from PIL import Image
import warnings


class MultimodalInputDataGenerator(IDG):
    """Modified version of ImageDataGenerator and DataframeIterator to yield both image numpy arrays and image captions from dataframe 
        
        
        Raises:
            ImportError -- must import pandas
            ValueError -- input column names of dataframe must be strings
            ValueError -- has_ext argument
            ValueError -- [description]
            ValueError -- [description]
            ValueError -- [description]
            ValueError -- [description]
            TypeError -- [description]

    """

    def __init__(self,num_patch_along_one_dimension=1,*args,**kwargs):
        
        self.bag_size =num_patch_along_one_dimension **2
        super(MultimodalInputDataGenerator,self).__init__(args,kwargs)
    

    def flow_from_dataframe(self, dataframe, directory,
                            x_col=["filename"], y_col="class", has_ext=True,
                            target_size=(256, 256), color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            subset=None,
                            interpolation='nearest',
                            sort=True,
                            follow_links=True,
                            drop_duplicates=False,
                            **kwargs):
        """Takes the dataframe and the path to a directory
         and generates batches of augmented/normalized data.
        # A simple tutorial can be found at: http://bit.ly/keras_flow_from_dataframe
        # Arguments
            dataframe: Pandas dataframe containing the filenames of the
                images in a column and classes in another or column/s
                that can be fed as raw target data.
            directory: string, path to the target directory that contains all
                the images mapped in the dataframe.
            x_col: string, column in the dataframe that contains
                the filenames of the target images.
            y_col: string or list of strings,columns in
                the dataframe that will be the target data.
            has_ext: bool, True if filenames in dataframe[x_col]
                has filename extensions,else False.
            target_size: tuple of integers `(height, width)`, default: `(256, 256)`.
                The dimensions to which all images found will be resized.
            color_mode: one of "grayscale", "rgb". Default: "rgb".
                Whether the images will be converted to have 1 or 3 color channels.
            classes: optional list of classes (e.g. `['dogs', 'cats']`).
                Default: None. If not provided, the list of classes will be
                automatically inferred from the `y_col`,
                which will map to the label indices, will be alphanumeric).
                The dictionary containing the mapping from class names to class
                indices can be obtained via the attribute `class_indices`.
            class_mode: one of "categorical", "binary", "sparse",
                "input", "other" or None. Default: "categorical".
                Determines the type of label arrays that are returned:
                - `"categorical"` will be 2D one-hot encoded labels,
                - `"binary"` will be 1D binary labels,
                - `"sparse"` will be 1D integer labels,
                - `"input"` will be images identical
                    to input images (mainly used to work with autoencoders).
                - `"other"` will be numpy array of `y_col` data
                - None, no labels are returned (the generator will only
                    yield batches of image data, which is useful to use
                `model.predict_generator()`, `model.evaluate_generator()`, etc.).
            batch_size: size of the batches of data (default: 32).
            shuffle: whether to shuffle the data (default: True)
            seed: optional random seed for shuffling and transformations.
            save_to_dir: None or str (default: None).
                This allows you to optionally specify a directory
                to which to save the augmented pictures being generated
                (useful for visualizing what you are doing).
            save_prefix: str. Prefix to use for filenames of saved pictures
                (only relevant if `save_to_dir` is set).
            save_format: one of "png", "jpeg"
                (only relevant if `save_to_dir` is set). Default: "png".
            follow_links: whether to follow symlinks inside class subdirectories
                (default: False).
            subset: Subset of data (`"training"` or `"validation"`) if
                `validation_split` is set in `ImageDataGenerator`.
            interpolation: Interpolation method used to resample the image if the
                target size is different from that of the loaded image.
                Supported methods are `"nearest"`, `"bilinear"`, and `"bicubic"`.
                If PIL version 1.1.3 or newer is installed, `"lanczos"` is also
                supported. If PIL version 3.4.0 or newer is installed, `"box"` and
                `"hamming"` are also supported. By default, `"nearest"` is used.
            sort: Boolean, whether to sort dataframe by filename (before shuffle).
            drop_duplicates: Boolean, whether to drop duplicate rows
                based on filename.
        # Returns
            A DataFrameIterator yielding tuples of `(x, y)`
            where `x` is a numpy array containing a batch
            of images with shape `(batch_size, *target_size, channels)`
            and `y` is a numpy array of corresponding labels.
        """

        return DataFramewithMultiModalInputIterator(dataframe, directory, self,
                                 x_cols=x_col, y_col=y_col, has_ext=has_ext,
                                 target_size=target_size, color_mode=color_mode,
                                 classes=classes, class_mode=class_mode,
                                 data_format=self.data_format,
                                 batch_size=batch_size, shuffle=shuffle, seed=seed,
                                 save_to_dir=save_to_dir,
                                 save_prefix=save_prefix,
                                 save_format=save_format,
                                 subset=subset,
                                 interpolation=interpolation,
                                 sort=sort,
                                 drop_duplicates=drop_duplicates,
                                 follow_links= follow_links,
                                 **kwargs)


class DataFramewithMultiModalInputIterator(Iterator):
    """Iterator capable of reading images from a directory on disk
    through a dataframe.
    # Arguments
    dataframe: Pandas dataframe containing the filenames of the
                images in a column and classes in another or column/s
                that can be fed as raw target data.
    directory: Path to the directory to read images from.
        Each subdirectory in this directory will be
        considered to contain images from one class,
        or alternatively you could specify class subdirectories
        via the `classes` argument.
        if used with dataframe,this will be the directory to under which
        all the images are present.
        You could also set it to None if data in x_col column are
        absolute paths.
    image_data_generator: Instance of `ImageDataGenerator`
        to use for random transformations and normalization.
    x_col: Column in dataframe that contains all the filenames (or absolute
        paths, if directory is set to None).
    y_col: Column/s in dataframe that has the target data.
    has_ext: bool, Whether the filenames in x_col has extensions or not.
    target_size: tuple of integers, dimensions to resize input images to.
    color_mode: One of `"rgb"`, `"rgba"`, `"grayscale"`.
        Color mode to read images.
    classes: Optional list of strings, names of
        each class (e.g. `["dogs", "cats"]`).
        It will be computed automatically if not set.
    class_mode: Mode for yielding the targets:
        `"binary"`: binary targets (if there are only two classes),
        `"categorical"`: categorical targets,
        `"sparse"`: integer targets,
        `"input"`: targets are images identical to input images (mainly
            used to work with autoencoders),
        `"other"`: targets are the data(numpy array) of y_col data
        `None`: no targets get yielded (only input images are yielded).
    batch_size: Integer, size of a batch.
    shuffle: Boolean, whether to shuffle the data between epochs.
    seed: Random seed for data shuffling.
    data_format: String, one of `channels_first`, `channels_last`.
    save_to_dir: Optional directory where to save the pictures
        being yielded, in a viewable format. This is useful
        for visualizing the random transformations being
        applied, for debugging purposes.
    save_prefix: String prefix to use for saving sample
        images (if `save_to_dir` is set).
    save_format: Format to use for saving sample images
        (if `save_to_dir` is set).
    subset: Subset of data (`"training"` or `"validation"`) if
        validation_split is set in ImageDataGenerator.
    interpolation: Interpolation method used to resample the image if the
        target size is different from that of the loaded image.
        Supported methods are "nearest", "bilinear", and "bicubic".
        If PIL version 1.1.3 or newer is installed, "lanczos" is also
        supported. If PIL version 3.4.0 or newer is installed, "box" and
        "hamming" are also supported. By default, "nearest" is used.
    sort: Boolean, whether to sort dataframe by filename (before shuffle).
    drop_duplicates: Boolean, whether to drop duplicate rows based on filename.
    """

    def __init__(self, dataframe, directory, image_data_generator,
            x_cols=["filenames","image_captions"], y_col="class", has_ext=True,
            target_size=(256, 256), color_mode='rgb',
            classes=None, class_mode='categorical',
            batch_size=32, shuffle=True, seed=None,
            data_format=None,
            save_to_dir=None, save_prefix='', save_format='png',
            follow_links=False,
            subset=None,
            interpolation='nearest',
            dtype='float32',
            sort=True,
            drop_duplicates=True,
            cap_token_vocab=None,
            num_tokens=None):
        
        
        try:
            super(DataFramewithMultiModalInputIterator, self).common_init(image_data_generator,
                                                        target_size,
                                                        color_mode,
                                                        data_format,
                                                        save_to_dir,
                                                        save_prefix,
                                                        save_format,
                                                        subset,
                                                        interpolation)
        except AttributeError:
            super().common_init(image_data_generator,      target_size,
                                                    color_mode,
                                                    data_format,
                                                    save_to_dir,
                                                    save_prefix,
                                                    save_format,
                                                    subset,
                                                    interpolation)
        try:
            import pandas as pd
        except ImportError:
            raise ImportError('Install pandas to use flow_from_dataframe.')
        if type(x_cols[0]) != str:
            raise ValueError("elements of x_cols  must be a string.")
        if len(x_cols)>1 and  type(x_cols[1]) != str :
            raise ValueError("elements of x_cols  must be a string.")
        


        if type(has_ext) != bool:
            raise ValueError("has_ext must be either True if filenames in"
                                " x_col has extensions,else False.")
        self.df = dataframe.copy()
        if y_col == "":
            self.df.assign(placeholder = pd.Series("classname",index=dataframe.index))
            y_col = "placeholder"


        if drop_duplicates:
            self.df.drop_duplicates(subset=x_cols+[y_col], inplace=True)
        self.x_col = x_cols[0]
        self.x_all_cols = x_cols
        for col in x_cols:
            self.df[col] = self.df[col].astype(str)
        self.num_tokens = num_tokens
        self.caption_token_vocab = cap_token_vocab
        self.directory = directory

        
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse',
                                'input', 'other', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                                '; expected one of "categorical", '
                                '"binary", "sparse", "input"'
                                '"other" or None.')
        self.class_mode = class_mode
        self.dtype = dtype
        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp',
                                'ppm', 'tif', 'tiff'}
        # First, count the number of samples and classes.
        self.samples = 0

        if not classes:
            classes = []
            if class_mode not in ["other", "input", None]:
                classes = list(self.df[y_col].unique())
        else:
            if class_mode in ["other", "input", None]:
                raise ValueError('classes cannot be set if class_mode'
                                    ' is either "other" or "input" or None.')
        self.num_classes = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        # Second, build an index of the images.
        self.filenames = []
        self.classes = np.zeros((self.samples,), dtype='int32')

        if self.directory is not None:
            filenames = _list_valid_filenames_in_directory(
                directory,
                white_list_formats,
                None,
                class_indices=self.class_indices,
                follow_links=follow_links,
                df=True)
                #df=False)
        else:
            if not has_ext:
                raise ValueError('has_ext cannot be set to False'
                                    ' if directory is None.')
            filenames = self._list_valid_filepaths(white_list_formats)

        if has_ext:
            ext_exist = False
            for ext in white_list_formats:
                if self.df[x_cols[0]].values[0].lower().endswith("." + ext):
                    ext_exist = True
                    break
            if not ext_exist:
                raise ValueError('has_ext is set to True but'
                                    ' extension not found in x_col')
            self.df = self.df[self.df[x_cols[0]].isin(filenames)]
            valid_image = np.full(self.df.shape[0],True)
            for df_i, imgpath in enumerate(self.df[x_cols[0]]):
                if df_i % 10000 == 0:
                    print("{}th image checked for file integrity".format(df_i))
                try:
                    img = Image.open(imgpath)
                    img.verify()
                except (IOError,SyntaxError,ValueError) as e:
                    print ('Bad file:' + str(imgpath))
                    valid_image[df_i] = False
            invalid_image = np.logical_not(valid_image)
            with open("invalid_image_list.txt",'w') as fh:
                for imgpath in self.df.loc[invalid_image,][x_cols[0]]:
                    fh.write(imgpath+"\n")

            self.df = self.df.loc[valid_image,]


            if sort:
                self.df.sort_values(by=x_cols[0], inplace=True)
            self.filenames = list(self.df[x_cols[0]])
        else:
            without_ext_with = {f[:-1 * (len(f.split(".")[-1]) + 1)]: f
                                for f in filenames}
            filenames_without_ext = [f[:-1 * (len(f.split(".")[-1]) + 1)]
                                        for f in filenames]
            self.df = self.df[self.df[x_cols[0]].isin(filenames_without_ext)]
            if sort:
                self.df.sort_values(by=x_cols[0], inplace=True)
            self.filenames = [without_ext_with[f] for f in list(self.df[x_cols[0]])]
        
        if self.split:
            num_files = len(self.filenames)
            start = int(self.split[0] * num_files)
            stop = int(self.split[1] * num_files)
            self.df = self.df.iloc[start: stop, :]
            self.filenames = self.filenames[start: stop]

        if class_mode not in ["other", "input", None]:
            classes = self.df[y_col].values
            self.classes = np.array([self.class_indices[cls] for cls in classes])
        elif class_mode == "other" :
            self.data = self.df[y_col].values
            if type(y_col) == str:
                y_col = [y_col]
            if "object" in list(self.df[y_col].dtypes):
                raise TypeError("y_col column/s must be numeric datatypes.")
        
        self.samples = len(self.filenames)
        if self.num_classes > 0:
            print('Found %d images belonging to %d classes.' %
                    (self.samples, self.num_classes))
        else:
            print('Found %d images.' % self.samples)
        self.captions=None
        if len(x_cols)>1:
            self.captions= list(self.df[x_cols[1]])

        super(DataFramewithMultiModalInputIterator, self).__init__(self.samples,
                                                batch_size,
                                                shuffle,
                                                seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(
            (len(index_array),) + self.image_shape,
            dtype=self.dtype)
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            if self.directory is not None:
                img_path = os.path.join(self.directory, fname)
            else:
                img_path = fname
            try:
                img = load_img(img_path,
                           color_mode=self.color_mode,
                           target_size=self.target_size,
                           interpolation=self.interpolation)
            except Exception as e:
                print (e)
                print ("{} cannot be loaded. Remove from dataset ".format(img_path))
                if i>0:
                    fname_alt =self.filenames[index_array[i-1]]
                else:
                    fname_alt =self.filenames[index_array[i+1]]
                if self.directory is not None:
                    img_path = os.path.join(self.directory, fname_alt)
                else:
                    img_path = fname_alt
                img = load_img(img_path,
                           color_mode=self.color_mode,
                           target_size=self.target_size,
                           interpolation=self.interpolation)
            x = img_to_array(img, data_format=self.data_format)
            # Pillow images should be closed after `load_img`,
            # but not PIL images.
            if hasattr(img, 'close'):
                img.close()
            params = self.image_data_generator.get_random_transform(x.shape)
            x = self.image_data_generator.apply_transform(x, params)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e7),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        
        batch_z = np.zeros((len(batch_x), self.num_tokens+1),dtype= self.dtype)
        if self.captions is not None:
            for i, cap in enumerate([self.captions[j] for j in index_array]):
                img_cap_tokens = cap.strip().split()
                for z in img_cap_tokens:
                    token_idx = self.caption_token_vocab.get(z,self.num_tokens)
                    batch_z[i,token_idx] = 1.
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(self.dtype)
        elif self.class_mode == 'categorical':
            batch_y = np.zeros(
                (len(batch_x), self.num_classes),
                dtype=self.dtype)
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        elif self.class_mode == 'other':
            batch_y = self.data[index_array]
        elif self.captions is None:
            return [batch_x]
        else:
            return [batch_x,batch_z]

        if self.captions is None:
            return [batch_x], batch_y
        else:
            return [batch_x,batch_z], batch_y
    def _list_valid_filepaths(self, white_list_formats):

        def get_ext(filename):
            return os.path.splitext(filename)[1][1:].lower()

        df_paths = self.df[self.x_col]

        format_check = df_paths.map(get_ext).isin(white_list_formats)
        existence_check = df_paths.map(os.path.isfile)

        valid_filepaths = list(df_paths[np.logical_and(format_check,
                                                       existence_check)])

        return valid_filepaths


def _iter_valid_files(directory, white_list_formats, follow_links):
    """Iterates on files with extension in `white_list_formats` contained in `directory`.
    # Arguments
        directory: Absolute path to the directory
            containing files to be counted
        white_list_formats: Set of strings containing allowed extensions for
            the files to be counted.
        follow_links: Boolean.
    # Yields
        Tuple of (root, filename) with extension in `white_list_formats`.
    """
    def _recursive_list(subpath):
        return sorted(os.walk(subpath, followlinks=follow_links),
                      key=lambda x: x[0])

    for root, _, files in _recursive_list(directory):
        for fname in sorted(files):
            for extension in white_list_formats:
                if fname.lower().endswith('.tiff'):
                    warnings.warn('Using \'.tiff\' files with multiple bands '
                                  'will cause distortion. '
                                  'Please verify your output.')
                if fname.lower().endswith('.' + extension):
                    yield root, fname

def _list_valid_filenames_in_directory(directory, white_list_formats, split,
                                       class_indices, follow_links, df=False):
    """Lists paths of files in `subdir` with extensions in `white_list_formats`.
    # Arguments
        directory: absolute path to a directory containing the files to list.
            The directory name is used as class label
            and must be a key of `class_indices`.
        white_list_formats: set of strings containing allowed extensions for
            the files to be counted.
        split: tuple of floats (e.g. `(0.2, 0.6)`) to only take into
            account a certain fraction of files in each directory.
            E.g.: `segment=(0.6, 1.0)` would only account for last 40 percent
            of images in each directory.
        class_indices: dictionary mapping a class name to its index.
        follow_links: boolean.
        df: boolean
    # Returns
        classes: a list of class indices(returns only if `df=False`)
        filenames: if `df=False`,returns the path of valid files in `directory`,
            relative from `directory`'s parent (e.g., if `directory` is
            "dataset", the filenames will be
            `["dataset/class1/file1.jpg", "dataset/class1/file2.jpg", ...]`).
            if `df=True`, returns the path of valid files in `directory`,
            relative from `directory` (e.g., if `directory` is
            "dataset", the filenames will be
            `["class1/file1.jpg", "class1/file2.jpg", ...]`).
    """
    dirname = os.path.basename(directory)
    if split:
        num_files = len(list(
            _iter_valid_files(directory, white_list_formats, follow_links)))
        start, stop = int(split[0] * num_files), int(split[1] * num_files)
        valid_files = list(
            _iter_valid_files(
                directory, white_list_formats, follow_links))[start: stop]
    else:
        valid_files = _iter_valid_files(
            directory, white_list_formats, follow_links)
    if df:
        filenames = []
        for root, fname in valid_files:
            absolute_path = os.path.join(root, fname)
            relative_path = os.path.relpath(absolute_path, directory)
            filenames.append(relative_path)
        return filenames
    classes = []
    filenames = []
    for root, fname in valid_files:
        classes.append(class_indices[dirname])
        absolute_path = os.path.join(root, fname)
        relative_path = os.path.join(
            dirname, os.path.relpath(absolute_path, directory))
        filenames.append(relative_path)

    return classes, filenames




