
class Load_BCIC_2a():
    '''
    Subclass of LoadData for loading BCI Competition IV Dataset 2a.

    Methods:
        get_epochs(self, tmin=-0., tmax=2, baseline=None, downsampled=None)
    '''
    def __init__(self, data_path, persion):
        self.stimcodes_train=('769','770','771','772')
        self.stimcodes_test=('783')
        self.data_path = data_path
        self.persion = persion
        self.channels_to_remove = ['EOG-left', 'EOG-central', 'EOG-right']
        super(Load_BCIC_2a,self).__init__()

    def get_epochs_train(self, tmin=-0., tmax=2, low_freq=None, high_freq=None, baseline=None, downsampled=None):
        file_to_load = 's{:}/'.format(self.persion) + 'A0' + self.persion + 'T.gdf'
        raw_data = mne.io.read_raw_gdf(self.data_path + file_to_load, preload=True)
        if low_freq and high_freq:
            raw_data.filter(l_freq=low_freq, h_freq=high_freq)
        if downsampled is not None:
            raw_data.resample(sfreq=downsampled)
        self.fs = raw_data.info.get('sfreq')
        events, event_ids = mne.events_from_annotations(raw_data)
        stims =[value for key, value in event_ids.items() if key in self.stimcodes_train]
        epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=tmin, tmax=tmax, event_repeated='drop',
                            baseline=baseline, preload=True, proj=False, reject_by_annotation=False)
        epochs = epochs.drop_channels(self.channels_to_remove)
        self.y_labels = epochs.events[:, -1] - min(epochs.events[:, -1])
        self.x_data = epochs.get_data()*1e6
        eeg_data={'x_data':self.x_data[:, :, :-1],
                  'y_labels':self.y_labels,
                  'fs':self.fs}
        return eeg_data

    def get_epochs_test(self, tmin=-0., tmax=2, low_freq=None, high_freq=None, baseline=None, downsampled=None):
        file_to_load = 's{:}/'.format(self.persion) + 'A0' + self.persion + 'E.gdf'
        raw_data = mne.io.read_raw_gdf(self.data_path + file_to_load, preload=True)
        data_path_label = self.data_path + "true_labels/A0" + self.persion + "E.mat"
        mat_label = scio.loadmat(data_path_label)
        mat_label = mat_label['classlabel'][:,0]-1
        if (low_freq is not None) and (high_freq is not None):
            raw_data.filter(l_freq=low_freq, h_freq=high_freq)
        if downsampled is not None:
            raw_data.resample(sfreq=downsampled)
        self.fs = raw_data.info.get('sfreq')
        events, event_ids = mne.events_from_annotations(raw_data)
        stims =[value for key, value in event_ids.items() if key in self.stimcodes_test]
        epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=tmin, tmax=tmax, event_repeated='drop',
                            baseline=baseline, preload=True, proj=False, reject_by_annotation=False)
        epochs = epochs.drop_channels(self.channels_to_remove)
        self.y_labels = epochs.events[:, -1] - min(epochs.events[:, -1]) + mat_label
        self.x_data = epochs.get_data()*1e6
        eeg_data={'x_data':self.x_data[:, :, :-1],
                  'y_labels':self.y_labels,
                  'fs':self.fs}
        return eeg_data


##%%
def get_data(path, subject=None, LOSO=False, Transfer=False, trans_num=0, onLine_2a=False,  data_model='one_session', isStandard=True, data_type='2a'):
    # Define dataset parameters
    fs = 250          # sampling rate
    t1 = int(2*fs)    # start time_point
    t2 = int(6*fs)    # end time_point
    T = t2-t1         # length of the MI trial (samples or time_points)
 
    # Load and split the dataset into training and testing 
    if LOSO:
        # Loading and Dividing of the data set based on the 
        # 'Leave One Subject Out' (LOSO) evaluation approach. 
        X_train, y_train, X_test, y_test, X_train_trans, y_train_trans = load_data_LOSO(path, subject, data_model, Transfer, trans_num)
    elif onLine_2a:
        X_train, y_train = load_data_onLine2a(path, data_model)
        X_test = []
        y_test = []
    else:
        # Loading and Dividing of the data set based on the subject-specific (subject-dependent) approach.
        # In this approach, we used the same training and testing data as the original competition, 
        # i.e., trials in session 1 for training, and trials in session 2 for testing.  
        path = path + 's{:}/'.format(subject)
        if data_type == '2a':
            X_train, y_train = load_data_2a(path, subject, True)
            X_test, y_test = load_data_2a(path, subject, False)
        elif data_type == '2b':
            load_raw_data = Load_BCIC_2b(path, subject)
            eeg_data = load_raw_data.get_epochs_train(tmin=0., tmax=4.)
            X_train, y_train = eeg_data['x_data'], eeg_data['y_labels']
            eeg_data = load_raw_data.get_epochs_test(tmin=0., tmax=4.)
            X_test, y_test = eeg_data['x_data'], eeg_data['y_labels']

    # Prepare training data
    N_tr, N_ch, samples = X_train.shape 
    if  data_type == '2a':
        X_train = X_train[:, :, t1:t2]
        y_train = y_train-1

    # Prepare testing data 
    if onLine_2a == False:
        if  data_type == '2a':
            X_test = X_test[:, :, t1:t2]
            y_test = y_test-1

    if Transfer:
        X_train_trans = X_train_trans[:, :, t1:t2]
        y_train_trans = y_train_trans-1
    else:
        X_train_trans = []
        y_train_trans = []

    # Standardize the data
    if (isStandard == True):
        if Transfer:
            X_train, X_test, X_train_trans = standardize_data_trans(X_train, X_test, X_train_trans, N_ch)
        elif onLine_2a:
            X_train = standardize_data_onLine2a(X_train, N_ch)
        else:
            X_train, X_test = standardize_data(X_train, X_test, N_ch)

    return X_train, y_train, X_test, y_test, X_train_trans, y_train_trans


##%%
def cross_validate(x_data, y_label, kfold, data_seed=20230520):
    '''
    This version dosen't use early stoping.
    Arg:
        sub:Subject number.
        data_path:The data path of all subjects.
        augment(bool):Data augment.Take care that this operation will change the size in the temporal dimension.
        validation_rate:The percentage of validation data in the data to be divided. 
        data_seed:The random seed for shuffle the data.
    @author:Guangjin Liang
    '''

    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=data_seed)
    for split_train_index,split_validation_index in skf.split(x_data,y_label):
        split_train_x       = x_data[split_train_index]
        split_train_y       = y_label[split_train_index]
        split_validation_x  = x_data[split_validation_index]
        split_validation_y  = y_label[split_validation_index]

        split_train_x,split_train_y = torch.FloatTensor(split_train_x),torch.LongTensor(split_train_y).reshape(-1)
        split_validation_x,split_validation_y = torch.FloatTensor(split_validation_x),torch.LongTensor(split_validation_y).reshape(-1)
   
        split_train_dataset = TensorDataset(split_train_x,split_train_y)
        split_validation_dataset = TensorDataset(split_validation_x,split_validation_y)
    
        yield split_train_dataset,split_validation_dataset


##%%
def BCIC_DataLoader(x_train, y_train, batch_size=64, num_workers=1, shuffle=True):
    '''
    Cenerate the batch data.

    Args:
        x_train: data to be trained
        y_train: label to be trained
        batch_size: the size of the one batch
        num_workers: how many subprocesses to use for data loading
        shuffle: shuffle the data
    '''
    # 将数据转换为TensorDataset类型
    dataset  = TensorDataset(x_train, y_train)
    # 分割数据，生成batch
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    # 函数返回值
    return dataloader
