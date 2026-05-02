import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def run_decision_tree():
    print("\n" + "="*50)
    print("DECISION TREE - Predicting Outcomes")
    print("="*50)
    
    # Create sample data: predicting if someone buys a computer
    print("\nCreating sample dataset...")
    print("Predicting: Will a customer buy a computer?")
    
    data = {
        'Age': ['Young', 'Young', 'Middle', 'Old', 'Old', 'Old', 
                'Middle', 'Young', 'Young', 'Old', 'Young', 'Middle',
                'Middle', 'Old'],
        'Income': ['High', 'High', 'High', 'Medium', 'Low', 'Low',
                   'Low', 'Medium', 'Low', 'Medium', 'Medium', 'Medium',
                   'High', 'Medium'],
        'Student': ['No', 'No', 'No', 'No', 'Yes', 'Yes',
                    'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No',
                    'Yes', 'No'],
        'Credit': ['Fair', 'Excellent', 'Fair', 'Fair', 'Fair', 'Excellent',
                   'Excellent', 'Fair', 'Fair', 'Fair', 'Excellent', 'Excellent',
                   'Fair', 'Excellent'],
        'BuysComputer': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No',
                         'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes',
                         'Yes', 'No']
    }
    
    df = pd.DataFrame(data)
    print(f"\nDataset ({len(df)} customers):")
    print(df.to_string())
    
    # Convert text to numbers
    df_encoded = pd.get_dummies(df.drop('BuysComputer', axis=1))
    X = df_encoded
    y = df['BuysComputer']
    
    print("\nFeatures used for prediction:")
    for i, feature in enumerate(X.columns):
        print(f"  {i+1}. {feature}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train decision tree
    print("\nTraining Decision Tree...")
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
    clf.fit(X_train, y_train)
    
    # Test accuracy
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nModel Accuracy: {accuracy*100:.1f}%")
    
    # Show tree rules as text
    print("\nDecision Tree Rules:")
    tree_rules = export_text(clf, feature_names=list(X.columns))
    print(tree_rules)
    
    # Create visualization
    plt.figure(figsize=(20, 10))
    plot_tree(clf, feature_names=list(X.columns), class_names=['No', 'Yes'],
              filled=True, rounded=True, fontsize=10)
    plt.title('Decision Tree: Will Customer Buy Computer?', fontsize=16)
    
    plt.savefig('decision_tree.png', dpi=150, bbox_inches='tight')
    print("\nTree visualization saved as: decision_tree.png")
    plt.show()
    
    # Example prediction
    print("\nExample Prediction:")
    print("New customer: Young, Medium income, Student, Fair credit")
    sample = pd.DataFrame([[1, 0, 0, 0, 1, 0, 1, 0]], columns=X.columns)
    prediction = clf.predict(sample)[0]
    print(f"Prediction: Will buy computer? -> {prediction}")

if __name__ == "__main__":
    run_decision_tree()