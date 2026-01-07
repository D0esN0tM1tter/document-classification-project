from services.evaluation import process_all_documents, evaluate, grid_search_evaluation, print_evaluation_report
from services.fusion_service import MultimodalDocumentClassifier
import pandas as pd


if __name__== "__main__" :

    df = pd.read_csv("data/predictions_formatted.csv")

    evaluation_result = evaluate(
        labels_path="data/test_labels_template.csv" , 
        predictions_df=df
    )

    print_evaluation_report(evaluation_result)



