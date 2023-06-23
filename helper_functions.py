import os
import torch
import numpy as np
import pandas as pd
import librosa as lr
from PIL import Image
import sklearn
import xgboost
import catboost
import lightgbm
from nltk.tag import pos_tag
from nltk.corpus import stopwords, wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import scipy
import noisereduce as nr
from tqdm.auto import tqdm
import moviepy.editor as mp
from scipy.io import wavfile
import librosa.display as ld
from sklearn.metrics import *
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from timeit import default_timer as Timer
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import GridSearchCV

def build_spectogram(audio_path, plot_path, bar = False):
    """
    Build spectrograms from audio files and save them as PNG images.

    Parameters:
        audio_path (str): The path to the directory containing the audio files.
        plot_path (str): The path to the directory where the spectrogram images will be saved.
        bar (bool, optional): Whether to include a colorbar in the spectrogram images. Default is False.

    Returns:
        None

    Raises:
        OSError: If there is an error while creating directories or loading audio files.

    """
    folders = []
    for item in os.listdir(audio_path):
        item_path = os.path.join(audio_path, item)
        if os.path.isdir(item_path):
            folders.append(item)

    for folder in folders:
        item_list = os.listdir(audio_path + folder)
        os.makedirs(plot_path+'/'+folder)
        for item in item_list:
            music, rate = lr.load(audio_path+folder+'/'+item)
            stft = lr.feature.melspectrogram(y=music, sr=rate, n_mels=256)
            db = lr.amplitude_to_db(stft)
            fig, ax = plt.subplots()
            img = ld.specshow(db, x_axis='time', y_axis='log', ax=ax)
            plt.axis(False)
            if bar == True:
                fig.colorbar(img, ax=ax, format='%0.2f')
            a = item.replace('.wav', '.png')
            plt.savefig(plot_path+'/'+folder+'/'+a)

def performance(model, x_test, y_test):
    """
    Calculates and displays the performance metrics of a trained model.

    Parameters:
    -----------
    model : object
        The trained machine learning model.

    x_test : array-like of shape (n_samples, n_features)
        The input test data.

    y_test : array-like of shape (n_samples,)
        The target test data.

    Returns:
    --------
    None

    Prints:
    -------
    Model Performance:
        Classification report containing precision, recall, F1-score, and support for each class.
    Accuracy:
        The accuracy of the model on the test data.
    Confusion Matrix:
        A plot of the confusion matrix, showing the true and predicted labels for the test data.

    Example:
    --------
    >>> performance(model, x_test, y_test)
    """

    preds = model.predict(x_test)
    accuracy = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    print("                 Model Performance")
    print(report)
    print(f"Accuracy = {round(accuracy*100, 2)}%")
    matrix = confusion_matrix(y_test, preds)
    matrix_disp = ConfusionMatrixDisplay(matrix)
    matrix_disp.plot(cmap='Blues')
    plt.show()
    
class CustomDataset_CSVlabels(Dataset):
    """
    A PyTorch dataset for loading spectrogram images and their corresponding labels from a CSV file.

    Args:
        csv_file (str): Path to the CSV file containing the image file names and labels.
        img_dir (str): Root directory where the image files are stored.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. 
            E.g, ``transforms.RandomCrop`` for randomly cropping an image.

    Attributes:
        img_labels (DataFrame): A pandas dataframe containing the image file names and labels.
        img_dir (str): Root directory where the image files are stored.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. 
            E.g, ``transforms.RandomCrop`` for randomly cropping an image.
    
    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(index): Returns the image and label at the given index.

    Returns:
        A PyTorch dataset object that can be passed to a DataLoader for batch processing.
    """
    def __init__(self,csv_file, img_dir, transform=None) -> None:
        super().__init__()
        self.img_labels = pd.read_csv(csv_file)
        self.img_labels.drop(['Unnamed: 0'], axis=1, inplace=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.img_labels)
    
    def __getitem__(self, index):
        """
        Returns the image and label at the given index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the image and label.
        """
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[index,0])
        image = Image.open(img_path)
        image = image.convert("RGB")
        y_label = torch.tensor(int(self.img_labels.iloc[index,1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)

def Train_Loop(
        num_epochs:int,
        train_dataloader:torch.utils.data.DataLoader,
        test_dataloader:torch.utils.data.DataLoader,
        model:torch.nn.Module,
        optimizer:torch.optim.Optimizer,
        loss_function:torch.nn.Module,
        device:str
):
    """
    Trains a PyTorch model using the given train and test dataloaders for the specified number of epochs.

    Parameters:
    -----------
    num_epochs : int
        The number of epochs to train the model for.
    train_dataloader : torch.utils.data.DataLoader
        The dataloader for the training data.
    test_dataloader : torch.utils.data.DataLoader
        The dataloader for the test/validation data.
    model : torch.nn.Module
        The PyTorch model to be trained.
    optimizer : torch.optim.Optimizer
        The optimizer to be used during training.
    loss_function : torch.nn.Module
        The loss function to be used during training.

    Returns:
    --------
    None

    Raises:
    -------
    None

    Notes:
    ------
    This function loops over the specified number of epochs and for each epoch, it trains the model on the training
    data and evaluates the performance on the test/validation data. During each epoch, it prints the training loss
    and the test loss and accuracy. At the end of training, it prints the total time taken for training.
    """
    model.to(device)
    start_time = Timer()
    
    for epoch in tqdm(range(num_epochs)):
        print(f"Epoch: {epoch}\n-----------")
        train_loss = 0
        for batch, (x,y) in enumerate(train_dataloader):
            x,y = x.to(device), y.to(device)
            y=y.float().squeeze()
            model.train()
            y_logits = model(x).squeeze()
            y_pred = torch.round(torch.sigmoid(y_logits))
            loss = loss_function(y_logits, y)
            train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if batch % 10 == 0:
            #     print(f"Looked at {batch * len(x)}/{len(train_dataloader.dataset)} samples")

        train_loss /= len(train_dataloader)
        
        test_loss, test_acc = 0, 0 
        test_log_loss = 0
        model.eval()
        with torch.inference_mode():
            for X, y in test_dataloader:
                X,y = X.to(device), y.to(device)
                y = y.float().squeeze()
                test_logits = model(X).squeeze()
                test_pred = torch.round(torch.sigmoid(test_logits))
                test_loss += loss_function(test_logits, y)
                test_acc += accuracy_score(y_true=y, y_pred=test_pred)
            test_loss /= len(test_dataloader)
            test_acc /= len(test_dataloader)
        print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc*100:.2f}%\n")

    end_time = Timer()
    print(f"Time taken = {end_time-start_time}")

class CustomDataset_FolderLabels:
    """
    CustomDataset class for loading and splitting a dataset into training, validation, and testing sets.

    Args:
        data_path (str): Path to the main folder containing subfolders for each class.
        train_ratio (float): Ratio of data allocated for the training set (0.0 to 1.0).
        val_ratio (float): Ratio of data allocated for the validation set (0.0 to 1.0).
        test_ratio (float): Ratio of data allocated for the testing set (0.0 to 1.0).
        batch_size (int): Number of samples per batch in the data loaders.
        transform (torchvision.transforms.transforms.Compose): Transformations to be applied on the image

    Attributes:
        train_loader (torch.utils.data.DataLoader): Data loader for the training set.
        val_loader (torch.utils.data.DataLoader): Data loader for the validation set.
        test_loader (torch.utils.data.DataLoader): Data loader for the testing set.

    """
    def __init__(self, data_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, batch_size=32, transform=None):
        self.data_path = data_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        if transform == None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transform
        self._load_dataset()

    def _load_dataset(self):
        """
        Loads the dataset and splits it into training, validation, and testing sets.

        """
        dataset = ImageFolder(root=self.data_path, transform=self.transform)
        num_samples = len(dataset)

        train_size = int(self.train_ratio * num_samples)
        val_size = int(self.val_ratio * num_samples)
        test_size = num_samples - train_size - val_size

        self.train_set, self.val_set, self.test_set = random_split(dataset, [train_size, val_size, test_size])

        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_set, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True)

    def get_train_loader(self):
        """
        Get the data loader for the training set.

        Returns:
            torch.utils.data.DataLoader: Data loader for the training set.

        """
        return self.train_loader

    def get_val_loader(self):
        """
        Get the data loader for the validation set.

        Returns:
            torch.utils.data.DataLoader: Data loader for the validation set.

        """
        return self.val_loader

    def get_test_loader(self):
        """
        Get the data loader for the testing set.

        Returns:
            torch.utils.data.DataLoader: Data loader for the testing set.

        """
        return self.test_loader
    

def balanced_log_loss(y_true, y_pred, x_test):
    """
    Compute the balanced logarithmic loss.

    Parameters:
        y_true : array-like of shape (n_samples,)
            True class labels.

        y_pred : array-like of shape (n_samples, n_classes)
            Predicted probabilities for each class.

        x_test : array-like of shape (n_samples, n_features)
            Test data used to calculate sample weights.

    Returns:
        balanced_log_loss : float
            The balanced logarithmic loss.

    Notes:
        The balanced logarithmic loss is computed by applying class weights to the
        log loss calculation, where the class weights are based on the distribution
        of class labels in the training data.

        The function first computes sample weights using the 'balanced' strategy,
        which assigns higher weights to minority classes and lower weights to majority
        classes. The sample weights are calculated based on the training data distribution
        of class labels.

        The log loss is then calculated using the true class labels (y_true), predicted
        probabilities (y_pred), and the computed sample weights.

        The balanced logarithmic loss is useful for evaluating classification models
        in imbalanced class scenarios, where each class is roughly equally important
        for the final score.

    Example:
        y_true = [0, 1, 0, 1, 1]
        y_pred = [[0.9, 0.1], [0.3, 0.7], [0.6, 0.4], [0.2, 0.8], [0.1, 0.9]]
        x_test = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]

        loss = balanced_log_loss(y_true, y_pred, x_test)
        print(f"Balanced Log Loss: {loss:.5f}")
    """
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_true)
    balanced_log_loss = log_loss(y_true, y_pred, sample_weight=sample_weights)
    return balanced_log_loss

def compare_performance(models:dict, x_test:np.ndarray, y_test:np.ndarray):
    """
    Compare the performance of multiple models on test data.

    Parameters:
        models (dict): A dictionary of model names as keys and corresponding trained models as values.
        x_test (np.ndarray): Test data for evaluation.
        y_test (np.ndarray): True labels for the test data.

    Returns:
        pd.DataFrame: A DataFrame containing the performance metrics (accuracy, precision, recall, f1-score and balanced log loss) for each model.

    """
    names = []
    accuracy = []
    f1 = {}
    f1[0] = []
    f1[1] = []
    recall = {}
    recall[0] = []
    recall[1] = []
    precision = {}
    precision[0] = []
    precision[1] = []
    loss = []
    results = pd.DataFrame(columns=['Name', 'Accuracy', 'Precision_0', 'Precision_1', 'Recall_0', 'Recall_1', 'f1-score_0', 'f1-score_1'])
    for key,value in models.items():
        names.append(key)

        accuracy.append(round(accuracy_score(y_test, value.predict(x_test)), 3))

        f1[0].append(round(f1_score(y_test, value.predict(x_test), average=None)[0], 3))
        f1[1].append(round(f1_score(y_test, value.predict(x_test), average=None)[1], 3))

        recall[0].append(round(recall_score(y_test, value.predict(x_test), average=None)[0], 3))
        recall[1].append(round(recall_score(y_test, value.predict(x_test), average=None)[1], 3))

        precision[0].append(round(precision_score(y_test, value.predict(x_test), average=None)[0], 3))
        precision[1].append(round(precision_score(y_test, value.predict(x_test), average=None)[1], 3))

    results['Name'] = names
    results['Accuracy'] = accuracy
    results['Precision_0'] = precision[0]
    results['Precision_1'] = precision[1]
    results['Recall_0'] = recall[0]
    results['Recall_1'] = recall[1]
    results['f1-score_0'] = f1[0]
    results['f1-score_1'] = f1[1]
    
    return results
  

LogisticRegression_param_grid = [
    {
    'C': [0.1, 0.2, 0.3, 0.5, 1],
    'penalty': ['l2'],
    'solver' : ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag', 'saga', 'liblinear', 'saga'],
    'max_iter' : [2000]
    },
    {
    'C': [0.1, 1, 10, 100],
    'penalty': ['l1'],
    'solver' : ['liblinear', 'saga'],
    'max_iter' : [2000]
    },
    {
    'C': [0.1, 1, 10, 100],
    'penalty': ['elasticnet'],
    'solver' : ['saga'],
    'l1_ratio': [0.2, 0.4, 0.6],
    'max_iter' : [2000]   
    }
]
  
DecisionTree_param_grid = [
  {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'splitter': ['best', 'random'],
    'max_features': ['sqrt', 'log2', None]
  }
]

KNN_param_grid = [
  {
    'n_neighbors': [5,7,10,15],
    'algorithm': ['ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [10,20,30,40,50],
    'metric': ['minkowski', 'cosine'],
    'n_jobs': [-1]
  }
]

SVC_param_grid = [
    {
    'C': [0.1, 0.4, 0.7, 1, 10],
    'gamma': ['scale', 'auto', 0.1, 0.4, 0.7, 1, 10],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'decision_function_shape' : ['ovo', 'ovr']
    }
]

AdaBoost_param_grid = [
  {
    'n_estimators': [30,50,75,100],
    'learning_rate': [0.5,0.75,1,2,5],
    'algorithm': ['SAMME', 'SAMME.R']
  }
]

GradientBoost_param_grid = [
  {
    'loss': ['log_loss', 'deviance', 'exponential'],
    'learning_rate': [0.05,0.1,0.2,0.5,1],
    'n_estimators': [50,75,100,150,200],
    'subsample': [0.4,0.6,0.8,1],
    'criterion': ['friedman_mse', 'squared_error'],
    'max_features': ['sqrt', 'log2', None]
  }
]

RandomForest_param_grid = [
  {
    'n_estimators': [50,75,100,150,200],
    'criterion': ['gini',' entropy', 'log_loss'],
    'max_depth': [5,7,10,None],
    'max_features': ['sqrt', 'log2', None],
    'n_jobs': [-1]
  }
]

HistGradientBoost_param_grid = [
  {
    'loss': ['auto', 'log_loss', 'binary_crossentropy', 'categorical_crossentropy'],
    'learning_rate': [0.05,0.1,0.2,0.5,1],
    'max_iter': [100,200,300,400],
    'l2_regularization': [0,0.2,0.4,0.6,8.0,1]
  }
]

XGBoostClassifier_param_grid = [
  {
    'max_depth': [3, 5, 7],  
    'learning_rate': [0.1, 0.01, 0.001], 
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 0.9, 1.0],  
    'colsample_bytree': [0.8, 0.9, 1.0]
  }
]

LGBMClassifier_param_grid = [
  {
    'max_depth': [3, 5, 7],  
    'learning_rate': [0.1, 0.01, 0.001], 
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 0.9, 1.0],  
    'colsample_bytree': [0.8, 0.9, 1.0]
  }
]

CatBoostClassifier_param_grid = [
  {
    'max_depth': [3, 5, 7],  
    'learning_rate': [0.1, 0.01, 0.001], 
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 0.9, 1.0],  
    'colsample_bytree': [0.8, 0.9, 1.0]
  }
]

def fix_y(y):
  """
    Fix the values in the target variable `y` such that all unique values are in ascending order.

    Parameters:
    -----------
    y : numpy.ndarray or array-like
        The target variable array containing the values to be fixed.

    Returns:
    --------
    numpy.ndarray
        The fixed target variable array with unique values in ascending order.

    Examples:
    ---------
    >>> y = [2, 1, 3, 2, 5, 4, 5, 3]
    >>> fix_y(y)
    array([1, 0, 2, 1, 4, 3, 4, 2])

    >>> y = np.array([3, 5, 2, 1, 4])
    >>> fix_y(y)
    array([2, 3, 1, 0, 2])
    """
  fixed_y = y.copy()
  unique = np.unique(y)
  if unique[0] != 0:
    fixed_y = np.where(fixed_y == unique[0], 0, fixed_y)
    unique[0] = 0
  for i in range(1,len(unique)):
    if unique[i]-unique[i-1] != 1:
      fixed_y = np.where(fixed_y == unique[i], unique[i-1]+1, fixed_y)
      unique[i] = unique[i-1]+1
  return fixed_y

def BestParam_search(models:dict, x, y):
  """
    Perform hyperparameter tuning using grid search for different models and print the best parameters and scores.

    Parameters:
    -----------
    models : dict
        A dictionary containing model names as keys and the corresponding model objects as values.
    x : array-like
        The feature matrix or input data.
    y : array-like
        The target variable or output data.

    Returns:
    --------
    None

    Examples:
    ---------
    >>> models = {
    ...     'Logistic Regression': LogisticRegression(),
    ...     'Decision Tree': DecisionTreeClassifier(),
    ...     'SVC': SVC()
    ... }
    >>> x = ...
    >>> y = ...
    >>> BestParam_search(models, x, y)
    For Model: Logistic Regression
    Best hyperparameters:  {'C': 1.0, 'penalty': 'l2'}
    Best score:  0.85

    For Model: Decision Tree
    Best hyperparameters:  {'criterion': 'gini', 'max_depth': 5}
    Best score:  0.78

    For Model: SVC
    Best hyperparameters:  {'C': 1.0, 'kernel': 'rbf'}
    Best score:  0.92
    """
  y = fix_y(y)
  for key,model in models.items():
    if isinstance(model, sklearn.linear_model._logistic.LogisticRegression):
      grid_search = GridSearchCV(model, LogisticRegression_param_grid, cv=5, scoring='accuracy')
      grid_search.fit(x,y)
      print(f"For Model: {key}")
      print("Best hyperparameters: ", grid_search.best_params_)
      print("Best score: ", grid_search.best_score_)

    elif isinstance(model, sklearn.tree._classes.DecisionTreeClassifier):
      grid_search = GridSearchCV(model, DecisionTree_param_grid, cv=5, scoring='accuracy')
      grid_search.fit(x,y)
      print(f"For Model: {key}")
      print("Best hyperparameters: ", grid_search.best_params_)
      print("Best score: ", grid_search.best_score_)

    elif isinstance(model, sklearn.svm._classes.SVC):
      grid_search = GridSearchCV(model, SVC_param_grid, cv=5, scoring='accuracy')
      grid_search.fit(x,y)
      print(f"For Model: {key}")
      print("Best hyperparameters: ", grid_search.best_params_)
      print("Best score: ", grid_search.best_score_)

    elif isinstance(model, sklearn.neighbors._classification.KNeighborsClassifier):
      grid_search = GridSearchCV(model, KNN_param_grid, cv=5, scoring='accuracy')
      grid_search.fit(x,y)
      print(f"For Model: {key}")
      print("Best hyperparameters: ", grid_search.best_params_)
      print("Best score: ", grid_search.best_score_)

    elif isinstance(model, lightgbm.sklearn.LGBMClassifier):
      grid_search = GridSearchCV(model, LGBMClassifier_param_grid, cv=5, scoring='accuracy')
      grid_search.fit(x,y)
      print(f"For Model: {key}")
      print("Best hyperparameters: ", grid_search.best_params_)
      print("Best score: ", grid_search.best_score_)

    elif isinstance(model, xgboost.sklearn.XGBClassifier):
      grid_search = GridSearchCV(model, XGBoostClassifier_param_grid, cv=5, scoring='accuracy')
      grid_search.fit(x,y)
      print(f"For Model: {key}")
      print("Best hyperparameters: ", grid_search.best_params_)
      print("Best score: ", grid_search.best_score_)

    elif isinstance(model, sklearn.ensemble._weight_boosting.AdaBoostClassifier):
      grid_search = GridSearchCV(model, AdaBoost_param_grid, cv=5, scoring='accuracy')
      grid_search.fit(x,y)
      print(f"For Model: {key}")
      print("Best hyperparameters: ", grid_search.best_params_)
      print("Best score: ", grid_search.best_score_)

    elif isinstance(model, sklearn.ensemble._gb.GradientBoostingClassifier):
      grid_search = GridSearchCV(model, GradientBoost_param_grid, cv=5, scoring='accuracy')
      grid_search.fit(x,y)
      print(f"For Model: {key}")
      print("Best hyperparameters: ", grid_search.best_params_)
      print("Best score: ", grid_search.best_score_)

    elif isinstance(model, sklearn.ensemble._forest.RandomForestClassifier):
      grid_search = GridSearchCV(model, RandomForest_param_grid, cv=5, scoring='accuracy')
      grid_search.fit(x,y)
      print(f"For Model: {key}")
      print("Best hyperparameters: ", grid_search.best_params_)
      print("Best score: ", grid_search.best_score_)

    elif isinstance(model, sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier):
      grid_search = GridSearchCV(model, HistGradientBoost_param_grid, cv=5, scoring='accuracy')
      grid_search.fit(x.toarray(),y)
      print(f"For Model: {key}")
      print("Best hyperparameters: ", grid_search.best_params_)
      print("Best score: ", grid_search.best_score_)

    else:
      continue

def get_wordnet_pos(treebank_tag:str):
    """
    Map a Treebank part-of-speech tag to the corresponding WordNet part-of-speech tag.

    Parameters:
    - treebank_tag (str): The Treebank part-of-speech tag.

    Returns:
    - str: The corresponding WordNet part-of-speech tag.

    Example:
    >>> get_wordnet_pos('NN')
    'n'

    The function takes a Treebank part-of-speech tag as input and returns the corresponding WordNet
    part-of-speech tag. It can be used to convert part-of-speech tags from Treebank format (used in
    NLTK, for example) to WordNet format.

    The mapping of Treebank tags to WordNet tags is as follows:
    - 'N' (Noun) -> 'n'
    - 'J' (Adjective) -> 'a'
    - 'V' (Verb) -> 'v'
    - 'R' (Adverb) -> 'r'
    - All other cases default to 'n' (Noun).

    Note that the returned WordNet tag is a single character string.

    Please refer to the NLTK documentation for more information about part-of-speech tagging and WordNet.
    """
    if treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN

class LemmTokenizer:
    """
    Tokenize and lemmatize text using NLTK's WordNetLemmatizer.

    Usage:
    tokenizer = LemmTokenizer()
    tokens = tokenizer("Example sentence")

    The LemmTokenizer class tokenizes and lemmatizes text using NLTK's WordNetLemmatizer. It provides a callable
    object, allowing it to be used as a function for tokenization and lemmatization.

    Methods:
    - __init__(): Initialize the LemmTokenizer object and create an instance of WordNetLemmatizer.
    - __call__(doc): Tokenize and lemmatize the input text.

    Example:
    >>> tokenizer = LemmTokenizer()
    >>> tokens = tokenizer("I am running in the park")
    >>> print(tokens)
    ['I', 'be', 'run', 'in', 'the', 'park']

    Note: This class requires NLTK and its dependencies to be installed.

    Please refer to the NLTK documentation for more information on tokenization, part-of-speech tagging, and lemmatization.
    """

    def __init__(self):
        """
        Initialize the LemmTokenizer object and create an instance of WordNetLemmatizer.
        """
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        """
        Tokenize and lemmatize the input text.

        Parameters:
        - doc (str): The text to be tokenized and lemmatized.

        Returns:
        - list: The list of lemmatized tokens.

        The __call__ method tokenizes the input text using word_tokenize and performs part-of-speech tagging using pos_tag.
        It then lemmatizes each token based on its part-of-speech tag using the WordNetLemmatizer and get_wordnet_pos functions.
        The resulting lemmatized tokens are returned as a list.
        """
        tokens = word_tokenize(doc)
        tokens_tags = pos_tag(tokens)
        return [self.wnl.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in tokens_tags]
