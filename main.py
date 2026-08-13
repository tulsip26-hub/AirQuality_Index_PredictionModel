from src.data_ingestion import data_loader
from src.data_preprocessing import preprocessing
from src.model_build import model_building

def main():
    
    df = data_loader()
    X_train,X_test,y_train,y_test = preprocessing(df)
    score = model_building(X_train,X_test,y_train,y_test)
    print(f'The Model Score is {score}') 
    
main()