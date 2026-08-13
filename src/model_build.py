# Import Machine Learning Building Model

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

def model_building(X_train,X_test,y_train,y_test):
    model = KNeighborsRegressor().fit(X_train,y_train) # Seen Data
    y_pred = model.predict(X_test)      # Unseen Data
    score = r2_score(y_test,y_pred)
    return score