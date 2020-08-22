from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.utils import resample


# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
#         data = data.drop(labels=self.columns, axis='columns')
#         columns = data.columns
#         x = data.values
#         min_max_scaler = preprocessing.MinMaxScaler()
#         x_scaled = min_max_scaler.fit_transform(x)
#         data = pd.DataFrame(x_scaled)
#         data.columns = columns
        
#         return data
        return data.drop(labels=self.columns, axis='columns')
    
