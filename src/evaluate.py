# src/evaluate.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

def evaluate_models(models_dict, X_test, y_test, output_dir="outputs"):
    """
    Evaluate multiple models and save:
        - Comparison table (CSV)
        - Confusion matrices (PNG)
        - Classification reports (TXT)
        - Accuracy bar chart (PNG)
    
    Args:
        models_dict (dict): {"model_name": trained_model}
        X_test, y_test: Test data
        output_dir (str): Folder to save outputs
    Returns:
        pd.DataFrame: Comparison table with Accuracy and F1-score
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        results.append({"Model": name, "Accuracy": acc, "F1-score": f1})
        
        # Save classification report
        report = classification_report(y_test, y_pred)
        with open(os.path.join(output_dir, f"classification_report_{name}.txt"), 'w') as f:
            f.write(report)
        
        # Save confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'confusion_matrix_{name}.png'))
        plt.close()
    
    # Create comparison table
    comparison_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
    comparison_df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)
    
    # Save Accuracy bar chart
    plt.figure(figsize=(8,5))
    plt.bar(comparison_df['Model'], comparison_df['Accuracy'], color='skyblue')
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0,1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"))
    plt.close()
    
    return comparison_df