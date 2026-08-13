# Import Data Peprocessing Libraries

from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.stats.mstats import winsorize

def preprocessing(df):
    
    df.drop_duplicates()

    for i in df.select_dtypes(exclude = 'object').columns:
        df[i] = winsorize(df[i],limits= (0.05,0.05))
        
    # Split the Dataset into X and y

    X = df.drop(columns = ['RH'],axis = 1)
    y = df['RH']

    # Split the dataset into train and test

    X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                    test_size = 0.3,
                                                    random_state = 1)

    # Using Scaling Technique

    sc = MinMaxScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train,X_test,y_train,y_test