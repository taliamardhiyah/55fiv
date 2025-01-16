import pandas as pd

def preprocess_data(data):
    df = pd.DataFrame(data['results'])
    df['category'] = df['number'].apply(lambda x: 'big' if x >= 5 else 'small')
    return df
