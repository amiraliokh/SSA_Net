# https://keras.io/examples/vision/3D_image_classification/#loading-data-and-preprocessing
from libraries import *
from utility_functions import crop_ct

class DatasetManager():
    
    def __normalize(self, volume):
        """Normalize the volume"""
        min = -1250
        max = 250
        volume[volume < min] = min
        volume[volume > max] = max
        volume = (volume - min) / (max - min) * 255
        volume = volume.astype("float32")
        return volume
    def __check_ct(self, ct_slice, lung_mask, infection_mask):
        """Check the volume for having infection segment in every slice"""
        to_be_deleted = []
        check_labels = np.array([1, 2])
        for j in range(infection_mask.shape[-1]):
            if True not in np.isin(check_labels, infection_mask[:, :, j]):
            # if not 1 in infection_mask[:, :, j] and not 2 in infection_mask[:, :, j]:
                to_be_deleted.append(j)
        infection_mask = np.delete(infection_mask, to_be_deleted, axis=-1)
        ct_slice = np.delete(ct_slice, to_be_deleted, axis=-1)
        lung_mask = np.delete(lung_mask, to_be_deleted, axis=-1)
        
        return [ct_slice, lung_mask, infection_mask]
    
    def __load_data(self, file_dir, n_class=1, normalizer=True, max_class:int=2):
        # recieves a dict containing ct scan, lung mask and infection mask file address
        ct_slice = nib.load(file_dir[0]).get_fdata()
        lung_mask = nib.load(file_dir[1]).get_fdata()
        infection_mask = nib.load(file_dir[2]).get_fdata()
        
        # check ct for having dimension equal or greater than 512 * 512 
        if ct_slice.shape[0] < 512 or ct_slice.shape[1] < 512:
            print(f'This file: {file_dir[0]}\n is not suitable for model training.')
            return
        
        # crop center of the image
        ct_slice, lung_mask, infection_mask = crop_ct(ct_slice, lung_mask, infection_mask)
        
        # remove slices that doesn't have any infection
        ct_slice, lung_mask, infection_mask = self.__check_ct(ct_slice, lung_mask, infection_mask)
        
        # normalize and transpose ct scan file
        if normalizer:
            ct_slice = self.__normalize(ct_slice)
            
        # remove left and right lung classification
        lung_mask[lung_mask == 2] = 1
        
        # remove any other infection class and classify as infection for single class
        # and remove selected classes such as plueral efusion
        infection_mask[infection_mask > max_class] = 0
        if n_class == 1:
            infection_mask[infection_mask > 1] = 1
            
        # remove slices that doesn't have any infection AGAIN!
        ct_slice, lung_mask, infection_mask = self.__check_ct(ct_slice, lung_mask, infection_mask)
        
        # reshape for having (n, h, w, c) shape
        infection_mask = np.expand_dims(np.transpose(infection_mask, (2, 0, 1)), -1)
        ct_slice = np.expand_dims(np.transpose(ct_slice, (2, 0, 1)), -1)
        lung_mask = np.expand_dims(np.transpose(lung_mask, (2, 0, 1)), -1)
        
        # one hot encoding for multiclass data
        if n_class != 1:
            infection_mask = K.squeeze(K.one_hot(K.cast(infection_mask, 'int32'), num_classes=n_class+1)[...], axis = 3)
        final_ct_slice = np.concatenate((ct_slice, lung_mask), -1).astype("float32")
        
        return [final_ct_slice, infection_mask]
    
    # save dataset 1
    def save_dataset_1(self, paths:list=[f'{CWD}/Data/Dataset 1/', ['CT scans/', 'Lung mask/', 'Infection mask/']], n_samples:int=21):
        # get file names of dataset one
        files_list = []
        path = paths[0]
        folders = os.listdir(path)
        for i, folder in enumerate(folders):
            direc = path + folder
            nii_files = os.listdir(direc)
            for _, file in enumerate(nii_files):
                if not file in files_list:
                    files_list.append(file)
        # save data to disk
        save_path = "Data/Dataset1_saved"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        x_filenames = []
        y_filenames = {}
        for index, file in enumerate(files_list[:n_samples]):
            file_dir = [
                paths[0] + paths[1][0] + file,
                paths[0] + paths[1][1] + file,
                paths[0] + paths[1][2] + file
            ]
            data = self.__load_data(file_dir, normalizer=(index < 9))

            
            if type(data) == list:
                start_index = len(x_filenames)
                for index, x in enumerate(data[0]):
                    x_filename = save_path + '/' + str(start_index + index) + 'x.npy'
                    y_filename = save_path + '/' + str(start_index + index) + 'y.npy'

                    x_filenames.append(x_filename)
                    y_filenames[x_filename] = y_filename

                    np.save(x_filename, x)
                    np.save(y_filename, data[1][index])

        with open(save_path + '/' + 'info.pkl', 'wb') as fp:
            pickle.dump([x_filenames, y_filenames, data[1][index].shape[-1]], fp)
        
        return [x_filenames, y_filenames, data[1][index].shape[-1]]

    # save and save dataset 2
    def save_dataset_2(self, path:str=f'{CWD}/Data/Dataset 2/'):
        # get file names of dataset 2
        files_list = []
        path = path
        nii_files = os.listdir(path)
        for _, file in enumerate(nii_files[:3]):
            if not file in files_list:
                files_list.append(file)
        # load ct scan files
        file_dir = [
            path + files_list[0],
            path + files_list[2],
            path + files_list[1]
        ]
        data = self.__load_data(file_dir, n_class=2, normalizer=True)

        # save data to disk
        save_path = "Data/Dataset2_saved"
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        x_filenames = []
        y_filenames = {}
        if type(data) == list:

            for index, x in enumerate(data[0]):
                x_filename = save_path + '/' + str(index) + 'x.npy'
                y_filename = save_path + '/' + str(index) + 'y.npy'

                x_filenames.append(x_filename)
                y_filenames[x_filename] = y_filename

                np.save(x_filename, x)
                np.save(y_filename, data[1][index])
        with open(save_path + '/' + 'info.pkl', 'wb') as fp:
            pickle.dump([x_filenames, y_filenames, data[1][index].shape[-1]], fp)

        return [x_filenames, y_filenames, data[1][index].shape[-1]]

    # save dataset 3
    def save_dataset_3(self, paths:list=[f'{CWD}/Data/Dataset 3/', ['CT Scans/', 'Lung mask/', 'Infection mask/']], n_samples:int=10):
        # get file names of dataset one
        files_list = []
        path = paths[0]
        folders = os.listdir(path)
        for i, folder in enumerate(folders):
            direc = path + folder
            nii_files = os.listdir(direc)
            for _, file in enumerate(nii_files):
                if not file in files_list:
                    files_list.append(file)
        # save data to disk
        save_path = "Data/Dataset3_saved"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        x_filenames = []
        y_filenames = {}
        for index, file in enumerate(files_list[:n_samples]):
            file_dir = [
                paths[0] + paths[1][0] + file,
                paths[0] + paths[1][1] + file,
                paths[0] + paths[1][2] + file
            ]
            data = self.__load_data(file_dir, n_class=2, normalizer=True)
            if type(data) == list:
                start_index = len(x_filenames)
                for index, x in enumerate(data[0]):
                    x_filename = save_path + '/' + str(start_index + index) + 'x.npy'
                    y_filename = save_path + '/' + str(start_index + index) + 'y.npy'

                    x_filenames.append(x_filename)
                    y_filenames[x_filename] = y_filename

                    np.save(x_filename, x)
                    np.save(y_filename, data[1][index])
        with open(save_path + '/' + 'info.pkl', 'wb') as fp:
            pickle.dump([x_filenames, y_filenames, data[1][index].shape[-1]], fp)
            
        return [x_filenames, y_filenames, data[1][index].shape[-1]]
    
    # load and save dataset 2
    def load_dataset_2(self, path:str=f'{CWD}/Data/Dataset2_saved/info.pkl'):
        # check if cleaned data is saved otherwise will clean and save data again
        if os.path.exists(path):
            with (open(path, "rb")) as openfile:
                 data = pickle.load(openfile)
        else: 
            data = self.save_dataset_2()
            
        return data
    
    # load and save dataset 1
    def load_dataset_1(self, path:str=f'{CWD}/Data/Dataset2_saved/info.pkl'):
        # check if cleaned data is saved otherwise will clean and save data again
        if os.path.exists(path):
            with (open(path, "rb")) as openfile:
                 data = pickle.load(openfile)
        else: 
            data = self.save_dataset_1()
            
        return data
    
    # load and save dataset 3
    def load_dataset_3(self, path:str=f'{CWD}/Data/Dataset2_saved/info.pkl'):
        # check if cleaned data is saved otherwise will clean and save data again
        if os.path.exists(path):
            with (open(path, "rb")) as openfile:
                 data = pickle.load(openfile)
        else: 
            data = self.save_dataset_3()
            
        return data
    
    # split data to default train and validation and test set
    def split_data(self, data:list, name:str='ssfs'):
        if name == 'ssfs':
            X_train, X_test, y_train, y_test = train_test_split(data[0], data[0], test_size=0.03, random_state=1)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.03, random_state=1)
        elif name == 'normal':
            X_train, X_test, y_train, y_test = train_test_split(data[0], data[0], test_size=0.1, random_state=1)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1)

        y_train = {key: value for key, value in data[1].items() if key in X_train}
        y_val = {key: value for key, value in data[1].items() if key in X_val}
        y_test = {key: value for key, value in data[1].items() if key in X_test}
    
        return (X_train, y_train, data[2]), (X_val, y_val, data[2]), (X_test, y_test, data[2])
    
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, label_length, batch_size=8, dim=(512, 512), n_channels=2,
                 shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.label_length = label_length
        self.shuffle = shuffle
        self.on_epoch_end()
        self.samples = len(list_IDs)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size,  *self.dim, self.label_length), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load(CWD + '/' + ID)

            # Store class
            y[i,] = np.load(self.labels[ID])

        return X, y