from services.fusion_service import MultimodalDocumentClassifier
import pandas as pd
import os
from pathlib import Path
from typing import Dict, List
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


def process_all_documents(directory_path: str, classifier: MultimodalDocumentClassifier) -> pd.DataFrame:
    """
    Process all PDF documents in a directory using the MultimodalDocumentClassifier.
    
    Args:
        directory_path: Path to directory containing PDF files
        classifier: Instance of MultimodalDocumentClassifier
        
    Returns:
        DataFrame with columns: filename, prediction, fused_score, nlp_score, vision_score
    """
    results = []
    
    # Get all PDF files in directory
    pdf_files = list(Path(directory_path).glob("*.pdf"))
    
    if not pdf_files:
        print(f"Warning: No PDF files found in {directory_path}")
        return pd.DataFrame(columns=['filename', 'prediction', 'fused_score', 'nlp_score', 'vision_score'])
    
    print(f"Processing {len(pdf_files)} documents...")
    
    for pdf_path in pdf_files:
        try:
            result = classifier.process_document(str(pdf_path))
            
            results.append({
                'filename': pdf_path.name,
                'prediction': result['prediction'],
                'fused_score': result['fused_score'],
                'nlp_score': result['nlp_result'].get('score', 0.0),
                'vision_score': result['vision_result'].get('score', 0.0)
            })
            
            print(f"✓ Processed: {pdf_path.name} -> {result['prediction']}")
            
        except Exception as e:
            print(f"✗ Error processing {pdf_path.name}: {str(e)}")
            results.append({
                'filename': pdf_path.name,
                'prediction': 'error',
                'fused_score': 0.0,
                'nlp_score': 0.0,
                'vision_score': 0.0
            })
    
    return pd.DataFrame(results)


def evaluate(labels_path: str, predictions_df: pd.DataFrame) -> Dict:
    """
    Evaluate predictions against ground truth labels.
    
    Args:
        labels_path: Path to CSV file with ground truth labels (columns: filename, label)
        predictions_df: DataFrame with predictions from process_all_documents
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Load ground truth labels
    labels_df = pd.read_csv(labels_path)
    
    # Merge predictions with labels
    merged_df = pd.merge(
        predictions_df, 
        labels_df, 
        on='filename', 
        how='inner'
    )
    
    if len(merged_df) == 0:
        print("Warning: No matching files between predictions and labels")
        return {}
    
    # Remove error predictions
    merged_df = merged_df[merged_df['prediction'] != 'error']
    
    y_true = merged_df['label'].values
    y_pred = merged_df['prediction'].values
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, pos_label='transaction', average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, pos_label='transaction', average='binary', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, pos_label='transaction', average='binary', zero_division=0),
        'support': len(y_true)
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=['transaction', 'non-transaction'])
    metrics['confusion_matrix'] = cm.tolist()
    
    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    metrics['classification_report'] = report
    
    return metrics


def grid_search_evaluation(
    test_directory: str,
    labels_path: str,
    fusion_weights: List[float] = [0.3, 0.5, 0.7],
    keyword_thresholds: List[int] = [3, 5, 7, 10],
    nlp_keywords_path: str = 'data/keywords.txt',
    vision_references_dir: str = 'data/references'
) -> pd.DataFrame:
    """
    Perform grid search over fusion weights and keyword thresholds.
    
    Args:
        test_directory: Directory containing test PDFs
        labels_path: Path to ground truth labels CSV
        fusion_weights: List of fusion weight values to test (weight for NLP)
        keyword_thresholds: List of keyword threshold values to test
        nlp_keywords_path: Path to NLP keywords file
        vision_references_dir: Path to vision references directory
        
    Returns:
        DataFrame with results for all parameter combinations
    """
    results = []
    
    total_combinations = len(fusion_weights) * len(keyword_thresholds)
    current = 0
    
    print(f"\nStarting grid search with {total_combinations} parameter combinations...")
    print(f"Fusion weights: {fusion_weights}")
    print(f"Keyword thresholds: {keyword_thresholds}")
    print("=" * 80)
    
    for fusion_weight in fusion_weights:
        for keyword_threshold in keyword_thresholds:
            current += 1
            
            print(f"\n[{current}/{total_combinations}] Testing: fusion_weight={fusion_weight}, keyword_threshold={keyword_threshold}")
            print("-" * 80)
            
            # Initialize classifier with current parameters
            classifier = MultimodalDocumentClassifier(
                nlp_keywords_path=nlp_keywords_path,
                nlp_keyword_threshold=keyword_threshold,
                vision_references_dir=vision_references_dir,
                fusion_weight_nlp=fusion_weight
            )
            
            # Process all documents
            predictions_df = process_all_documents(test_directory, classifier)
            
            # Evaluate
            metrics = evaluate(labels_path, predictions_df)
            
            if metrics:
                result = {
                    'fusion_weight_nlp': fusion_weight,
                    'keyword_threshold': keyword_threshold,
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1_score'],
                    'support': metrics['support']
                }
                
                results.append(result)
                
                print(f"Results: Accuracy={metrics['accuracy']:.3f}, Precision={metrics['precision']:.3f}, "
                      f"Recall={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}")
            else:
                print("Evaluation failed - no results")
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        # Find best configuration
        best_idx = results_df['f1_score'].idxmax()
        best_config = results_df.loc[best_idx]
        
        print("\n" + "=" * 80)
        print("GRID SEARCH COMPLETE")
        print("=" * 80)
        print(f"\nBest Configuration:")
        print(f"  Fusion Weight (NLP): {best_config['fusion_weight_nlp']}")
        print(f"  Keyword Threshold: {best_config['keyword_threshold']}")
        print(f"  F1 Score: {best_config['f1_score']:.4f}")
        print(f"  Accuracy: {best_config['accuracy']:.4f}")
        print(f"  Precision: {best_config['precision']:.4f}")
        print(f"  Recall: {best_config['recall']:.4f}")
    
    return results_df


def print_evaluation_report(metrics: Dict):
    """
    Print a formatted evaluation report.
    
    Args:
        metrics: Dictionary returned from evaluate()
    """
    if not metrics:
        print("No metrics to display")
        return
    
    print("\n" + "=" * 80)
    print("EVALUATION REPORT")
    print("=" * 80)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    print(f"  Support:   {metrics['support']}")
    
    print(f"\nConfusion Matrix:")
    cm = np.array(metrics['confusion_matrix'])
    print(f"                    Predicted")
    print(f"                    Transaction  Non-Transaction")
    print(f"Actual Transaction      {cm[0][0]:4d}         {cm[0][1]:4d}")
    print(f"       Non-Transaction  {cm[1][0]:4d}         {cm[1][1]:4d}")
    
    print(f"\nDetailed Classification Report:")
    report = metrics['classification_report']
    for label in ['transaction', 'non-transaction']:
        if label in report:
            print(f"\n  {label}:")
            print(f"    Precision: {report[label]['precision']:.4f}")
            print(f"    Recall:    {report[label]['recall']:.4f}")
            print(f"    F1-Score:  {report[label]['f1-score']:.4f}")
            print(f"    Support:   {report[label]['support']}")
    
    print("=" * 80)
