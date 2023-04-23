from libraries import *
# custom function for ploting model
def my_plot_model(model:Model):
    tf.keras.plot_model(
        model,                                     
        to_file='model.png', dpi=96,               
        show_shapes=True, show_layer_names=True,   
        expand_nested=False                        
    )
    
# crop ct scan volume and its masks(optional)
def crop_ct(ct_slice, lung_mask, infection_mask):
    if ct_slice.shape[0] != 512:
        if ct_slice.shape[0] % 2 == 1: 
            crop_len = int(((ct_slice.shape[0] - 1) - 512) / 2)
        else: 
            crop_len = int((ct_slice.shape[0] - 512) / 2)
        ct_slice = ct_slice[crop_len:(512 + crop_len), :, :]
        lung_mask = lung_mask[crop_len:(512 + crop_len), :, :]
        infection_mask = infection_mask[crop_len:(512 + crop_len), :, :]
    if ct_slice.shape[1] != 512:
        if ct_slice.shape[1] % 2 == 1: 
            crop_len = int(((ct_slice.shape[1] - 1) - 512) / 2)
        else: 
            crop_len = int((ct_slice.shape[1] - 512) / 2)
        ct_slice = ct_slice[:, crop_len:(512 + crop_len), :]
        lung_mask = lung_mask[:, crop_len:(512 + crop_len), :]
        infection_mask = infection_mask[:, crop_len:(512 + crop_len), :]
        
    return [ct_slice, lung_mask, infection_mask]
# Load model with spicified weights
def load_model(model:keras.Model, n_classes:int, weights:str):
    if n_classes == 1:
        model.build((None, 512, 512, n_classes))
    else:
        model.build((None, 512, 512, n_classes+1))
    
    model.load_weights(weights)
    return model
# Predict infection mask for a test ct-slice
def predict(ct_file_adr:str, lung_mask_file_adr:str, slice_no:int, model:keras.Model):
    ct_slice = nib.load(ct_file_adr).get_fdata()[slice_no]
    lung_mask = nib.load(lung_mask_file_adr).get_fdata()[slice_no]
    input = np.concatenate((ct_slice, lung_mask), axis = -1)
    output = model.predict(input)  
    if len(output) == 5:
        output = output[0]
    return output

def download_dataset_1():
    # Making directories
    if not os.path.exists('/content/Data/'):
        os.mkdir('/content/Data/')
    if not os.path.exists('/content/Data/Dataset 1/'):
        os.mkdir('/content/Data/Dataset 1/')
        
    # Download dataset 1
    URL = "https://zenodo.org/record/3757476/files/COVID-19-CT-Seg_20cases.zip?download=1"
    response = requests.get(URL)
    open("/content/Data/Dataset 1/ct_scans.zip", "wb").write(response.content)
    URL = "https://zenodo.org/record/3757476/files/Infection_Mask.zip?download=1"
    response = requests.get(URL)
    open("/content/Data/Dataset 1/infection_masks.zip", "wb").write(response.content)
    URL = "https://zenodo.org/record/3757476/files/Lung_Mask.zip?download=1"
    response = requests.get(URL)
    open("/content/Data/Dataset 1/lung_masks.zip", "wb").write(response.content)
    
    # Making directories
    if not os.path.exists('/content/Data/Dataset 1/CT scans/'):
        os.mkdir('/content/Data/Dataset 1/CT scans/')
    if not os.path.exists('/content/Data/Dataset 1/Lung mask/'):
        os.mkdir('/content/Data/Dataset 1/Lung mask/')
    if not os.path.exists('/content/Data/Dataset 1/Infection mask/'):
        os.mkdir('/content/Data/Dataset 1/Infection mask/')
        
    # Unzip downloaded files
    with zipfile.ZipFile('/content/Data/Dataset 1/ct_scans.zip', 'r') as zip_ref:
        zip_ref.extractall('/content/Data/Dataset 1/CT scans/')
        os.remove('/content/Data/Dataset 1/ct_scans.zip')  
        os.remove('/content/Data/Dataset 1/CT scans/ReadMe.txt')
    with zipfile.ZipFile('/content/Data/Dataset 1/infection_masks.zip', 'r') as zip_ref:
        zip_ref.extractall('/content/Data/Dataset 1/Lung mask/')
        os.remove('/content/Data/Dataset 1/infection_masks.zip')
    with zipfile.ZipFile('/content/Data/Dataset 1/lung_masks.zip', 'r') as zip_ref:
        zip_ref.extractall('/content/Data/Dataset 1/Infection mask/')
        os.remove('/content/Data/Dataset 1/lung_masks.zip')
        
    # Extract gz files and delete archives
    def gz_extract(directory):
        extension = ".gz"
        os.chdir(directory)
        for item in os.listdir(directory):
            if item.endswith(extension):
                gz_name = os.path.abspath(item)
                file_name = (os.path.basename(gz_name)).rsplit('.',1)[0]
                with gzip.open(gz_name,"rb") as f_in, open(file_name,"wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                os.remove(gz_name)

    gz_extract('/content/Data/Dataset 1/CT scans/')
    gz_extract('/content/Data/Dataset 1/Lung mask/')
    gz_extract('/content/Data/Dataset 1/Infection mask/')
    
def download_dataset_2():
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)
    # Making directories
    if not os.path.exists('/content/Data/'):
        os.mkdir('/content/Data/')
    if not os.path.exists('/content/Data/Dataset 2/'):
        os.mkdir('/content/Data/Dataset 2/')
    
    # Doanload data from google drive
    file_id = '1SJoMelgRqb0EuqlTuq6dxBWf2j9Kno8S' #<-- You add in here the id from you google drive file, you can find it
    download = drive.CreateFile({'id': file_id})
    download.GetContentFile('/content/Data/Dataset 2/ct_scans.nii.gz')
    file_id = '1MEqpbpwXjrLrH42DqDygWeSkDq0bi92f' #<-- You add in here the id from you google drive file, you can find it
    download = drive.CreateFile({'id': file_id})
    download.GetContentFile('/content/Data/Dataset 2/infection_masks.nii.gz')
    file_id = '1zj4N_KV0LBko1VSQ7FPZ38eaEGNU0K6-' #<-- You add in here the id from you google drive file, you can find it
    download = drive.CreateFile({'id': file_id})
    download.GetContentFile('/content/Data/Dataset 2/lung_masks.nii.gz')
    
    # Extract gzip files
    with gzip.open('/content/Data/Dataset 2/ct_scans.nii.gz', 'rb') as f_in:
        with open('/content/Data/Dataset 2/ct_scans.nii', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
            os.remove('/content/Data/Dataset 2/ct_scans.nii.gz')
    with gzip.open('/content/Data/Dataset 2/infection_masks.nii.gz', 'rb') as f_in:
        with open('/content/Data/Dataset 2/infection_masks.nii', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
            os.remove('/content/Data/Dataset 2/infection_masks.nii.gz')
    with gzip.open('/content/Data/Dataset 2/lung_masks.nii.gz', 'rb') as f_in:
        with open('/content/Data/Dataset 2/lung_masks.nii', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
            os.remove('/content/Data/Dataset 2/lung_masks.nii.gz')
            
def download_dataset_3():
    # Making directories
    if not os.path.exists('/content/Data/'):
        os.mkdir('/content/Data/')
    if not os.path.exists('/content/Data/Dataset 3/'):
        os.mkdir('/content/Data/Dataset 3/')
        
    # Download Dataset 3
    URL = "https://figshare.com/ndownloader/files/25954007"
    response = requests.get(URL)
    open("/content/Data/Dataset 3/ct_scans.zip", "wb").write(response.content)
    URL = "https://figshare.com/ndownloader/files/25954013"
    response = requests.get(URL)
    open("/content/Data/Dataset 3/infection_masks.zip", "wb").write(response.content)
    URL = "https://figshare.com/ndownloader/files/25954010"
    response = requests.get(URL)
    open("/content/Data/Dataset 3/lung_masks.zip", "wb").write(response.content)
    
    # Making directories
    if not os.path.exists('/content/Data/Dataset 3/CT scans/'):
        os.mkdir('/content/Data/Dataset 3/CT scans/')
    if not os.path.exists('/content/Data/Dataset 3/Lung mask/'):
        os.mkdir('/content/Data/Dataset 3/Lung mask/')
    if not os.path.exists('/content/Data/Dataset 3/Infection mask/'):
        os.mkdir('/content/Data/Dataset 3/Infection mask/')
        
    # Unzip downloaded files
    with zipfile.ZipFile('/content/Data/Dataset 3/ct_scans.zip', 'r') as zip_ref:
        zip_ref.extractall('/content/Data/Dataset 3/CT scans/')
        os.remove('/content/Data/Dataset 3/ct_scans.zip')
    with zipfile.ZipFile('/content/Data/Dataset 3/infection_masks.zip', 'r') as zip_ref:
        zip_ref.extractall('/content/Data/Dataset 3/Infection mask/')
        os.remove('/content/Data/Dataset 3/infection_masks.zip')
    with zipfile.ZipFile('/content/Data/Dataset 3/lung_masks.zip', 'r') as zip_ref:
        zip_ref.extractall('/content/Data/Dataset 3/Lung mask/')
        os.remove('/content/Data/Dataset 3/lung_masks.zip')
    
    source_dir = '/content/Data/Dataset 3/CT scans/rp_im'
    target_dir = '/content/Data/Dataset 3/CT scans/'
    file_names = os.listdir(source_dir)
    for file_name in file_names:
        shutil.move(os.path.join(source_dir, file_name), target_dir)
    os.rmdir(source_dir)

    source_dir = '/content/Data/Dataset 3/Lung mask/rp_lung_msk'
    target_dir = '/content/Data/Dataset 3/Lung mask/'
    file_names = os.listdir(source_dir)
    for file_name in file_names:
        shutil.move(os.path.join(source_dir, file_name), target_dir)
    os.rmdir(source_dir)

    source_dir = '/content/Data/Dataset 3/Infection mask/rp_msk'
    target_dir = '/content/Data/Dataset 3/Infection mask/'
    file_names = os.listdir(source_dir)
    for file_name in file_names:
        shutil.move(os.path.join(source_dir, file_name), target_dir)
    os.rmdir(source_dir)
        
    # Extract gz files and delete archives
    def gz_extract(directory):
        extension = ".gz"
        os.chdir(directory)
        for item in os.listdir(directory):
            if item.endswith(extension):
                gz_name = os.path.abspath(item)
                file_name = (os.path.basename(gz_name)).rsplit('.',1)[0]
                with gzip.open(gz_name,"rb") as f_in, open(file_name,"wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                os.remove(gz_name)

    gz_extract('/content/Data/Dataset 3/CT scans/')
    gz_extract('/content/Data/Dataset 3/Lung mask/')
    gz_extract('/content/Data/Dataset 3/Infection mask/')