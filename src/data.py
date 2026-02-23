
import seaborn as sns
from sklearn.model_selection import train_test_split

def load_data(features=None, target='survived', test_size=0.2, random_state=42):
    # Load Titanic dataset
    titanic = sns.load_dataset('titanic')
    
    if features is None:
        features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'class', 'who', 'adult_male', 'alone']
    
    X = titanic[features]
    y = titanic[target]
    
    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test