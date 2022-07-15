import pandas as pd

class ETL:

    def __init__(self):
        
        pass

    def load_csv(self, filename: str) -> pd.DataFrame:
        
        df=pd.read_csv(filename)
        return df

if __name__ == "__main__":
    etl = ETL()
    dataframe=etl.load_csv("/workspaces/Recomendador/data/ratings.csv")