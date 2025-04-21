import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# --- Pairwise comparison ---
def pairwiseComp(df, fuzz):
    print("\n🔍 Potential Duplicate Pairs:\n")
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            name_score = fuzz.ratio(df.loc[i, 'company_name_norm'], df.loc[j, 'company_name_norm'])
            domain_score = fuzz.ratio(df.loc[i, 'website_domain_norm'], df.loc[j, 'website_domain_norm'])

            # You can tweak these thresholds
            if name_score > 90 or domain_score > 95:
                print(f"🟡 Row {i} and Row {j} may be duplicates:")
                print(f"  → Name Similarity: {name_score}")
                print(f"  → Domain Similarity: {domain_score}")
                print(f"  → Names: {df.loc[i, 'company_name']} ↔ {df.loc[j, 'company_name']}")
                print(f"  → Domains: {df.loc[i, 'website_domain']} ↔ {df.loc[j, 'website_domain']}")
                print("—" * 50)


def barChart(top_countries, top_n):
    plt.figure(figsize=(12, 6))
    top_countries.plot(kind='bar', color='skyblue')
    plt.title(f"Top {top_n} Countries by Number of Companies")
    plt.xlabel("Country")
    plt.ylabel("Number of Companies")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def scatterPlot(df, reduced):
    vis_df = pd.DataFrame()
    vis_df['x'] = reduced[:, 0]
    vis_df['y'] = reduced[:, 1]
    vis_df['cluster'] = df['cluster_id']

    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=vis_df, x='x', y='y', hue='cluster', palette='hsv', legend=False, s=10)
    plt.title("Company Clusters (PCA-Reduced TF-IDF Features)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.tight_layout()
    plt.show()