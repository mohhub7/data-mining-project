import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

def run_clustering():
    print("\n" + "="*50)
    print("K-MEANS CLUSTERING - Grouping Similar Data")
    print("="*50)
    
    # Create sample data (like customer spending habits)
    print("\nGenerating sample customer data...")
    np.random.seed(42)
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)
    
    print(f"Created {len(X)} data points")
    print("Each point represents a customer with 2 features:")
    print("  - X axis: Annual income")
    print("  - Y axis: Spending score")
    
    # Run K-Means
    print("\nRunning K-Means with K=4 clusters...")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_
    
    print(f"\nCluster centers found:")
    for i, center in enumerate(centers):
        print(f"  Cluster {i+1}: ({center[0]:.1f}, {center[1]:.1f})")
    
    # Count points in each cluster
    unique, counts = np.unique(labels, return_counts=True)
    print("\nCustomers in each cluster:")
    for cluster, count in zip(unique, counts):
        print(f"  Cluster {cluster+1}: {count} customers")
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    colors = ['red', 'blue', 'green', 'purple']
    
    for i in range(4):
        cluster_points = X[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   c=colors[i], label=f'Cluster {i+1}', alpha=0.6)
    
    plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='X', 
                s=200, label='Centers')
    
    plt.title('Customer Segments (K-Means Clustering)', fontsize=14)
    plt.xlabel('Annual Income')
    plt.ylabel('Spending Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('clustering_result.png', dpi=150, bbox_inches='tight')
    print("\nChart saved as: clustering_result.png")
    plt.show()

if __name__ == "__main__":
    run_clustering()