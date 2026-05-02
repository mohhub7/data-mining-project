import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

def run_apriori():
    print("\n" + "="*50)
    print("APRIORI ALGORITHM - Finding Shopping Patterns")
    print("="*50)
    
    # Load data
    dataset = []
    with open('datasets/store_data.csv', 'r') as f:
        for line in f:
            items = line.strip().split(',')
            dataset.append(items)
    
    print(f"\nLoaded {len(dataset)} transactions")
    print("Sample transactions:")
    for i, transaction in enumerate(dataset[:3]):
        print(f"  Transaction {i+1}: {transaction}")
    
    # Convert to format Apriori needs
    te = TransactionEncoder()
    te_array = te.fit_transform(dataset)
    df = pd.DataFrame(te_array, columns=te.columns_)
    
    # Find frequent itemsets
    print("\nFinding frequent itemsets (min support = 30%)...")
    frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)
    
    if frequent_itemsets.empty:
        print("No frequent itemsets found.")
        return
    
    print("\nFrequent Itemsets:")
    for _, row in frequent_itemsets.iterrows():
        items = ', '.join(list(row['itemsets']))
        print(f"  {{{items}}} - appears in {row['support']*100:.0f}% of transactions")
    
    # Generate association rules
    print("\nGenerating association rules (min confidence = 50%)...")
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    
    if rules.empty:
        print("No strong rules found.")
        return
    
    print("\nAssociation Rules (IF...THEN...):")
    for _, rule in rules.iterrows():
        left = ', '.join(list(rule['antecedents']))
        right = ', '.join(list(rule['consequents']))
        print(f"\n  IF someone buys: {{{left}}}")
        print(f"  THEN they also buy: {{{right}}}")
        print(f"  Confidence: {rule['confidence']*100:.1f}%")

if __name__ == "__main__":
    run_apriori()