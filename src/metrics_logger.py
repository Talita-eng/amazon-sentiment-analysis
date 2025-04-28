import pandas as pd
from sklearn.metrics import classification_report
import datetime

def log_metrics(y_true, y_pred, file_path='metrics_log.csv'):
    report = classification_report(y_true, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df['timestamp'] = pd.to_datetime('now')
    
    # Salvar ou atualizar o arquivo de log
    try:
        existing_data = pd.read_csv(file_path)
        updated_data = pd.concat([existing_data, df], ignore_index=True)
    except FileNotFoundError:
        updated_data = df
    
    updated_data.to_csv(file_path, index=False)
