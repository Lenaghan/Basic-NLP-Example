def split_data(data, split_ratio = 0.8):
    from sklearn.model_selection import train_test_split
    '''
	Input:
		data: dataframe containing the dataset
		split_ratio: desired ratio of the train and test splits
		
	Output:
		train: train split of the data
		test: test split of the data
	'''
    train, test = train_test_split(data, test_size = 1 - split_ratio, shuffle=False)
    return train, test