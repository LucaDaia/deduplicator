import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from itertools import combinations
from sklearn.decomposition import PCA
from collections import defaultdict
from utils import pairwiseComp, barChart, scatterPlot

file_path = 'veridion_entity_resolution_challenge.snappy.parquet'
df = pd.read_parquet(file_path)

def normalize(text):
    if pd.isna(text):
        return ''
    return str(text).lower().strip()

df['company_name_norm'] = df['company_name'].apply(normalize)
df['website_domain_norm'] = df['website_domain'].apply(normalize)

print(df.shape)
print(df['main_country'].nunique())
print(df['main_country'].value_counts())

country_counts = df['main_country'].value_counts()
top_countries = country_counts.head(15)

barChart(top_countries, 15)


# pairwiseComp(df, fuzz)


# --------------------------VECTORIZARE--------------------------------
df['combined_text'] = df['company_name_norm'].astype(str) + ' ' + df['website_domain_norm'].astype(str)

vectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
X = vectorizer.fit_transform(df['combined_text'])

# k-means
n_clusters = 3000
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['cluster_id'] = kmeans.fit_predict(X)

cluster_counts = df['cluster_id'].value_counts()
duplicate_clusters = cluster_counts[cluster_counts > 1].index

# print("\nExemplu clustere:\n")
# for cluster in duplicate_clusters[:5]:
#     print(df[df['cluster_id'] == cluster][['company_name', 'website_domain', 'main_country']])


SIMILARITY_THRESHOLD = 90


duplicate_pairs = []
clusters = df[df['cluster_id'] != -1].groupby('cluster_id')

for cluster_id, group in clusters:
    rows = group.reset_index(drop=True)


    for i, j in combinations(range(len(rows)), 2):
        name1 = rows.loc[i, 'company_name_norm']
        name2 = rows.loc[j, 'company_name_norm']
        domain1 = rows.loc[i, 'website_domain_norm']
        domain2 = rows.loc[j, 'website_domain_norm']
        name_score = fuzz.ratio(name1, name2)
        domain_score = fuzz.ratio(domain1, domain2)


        if domain_score >= SIMILARITY_THRESHOLD and name_score >= SIMILARITY_THRESHOLD:
            duplicate_pairs.append({
                'cluster_id': cluster_id,
                'index_1': rows.index[i],
                'index_2': rows.index[j],
                'name_1': name1,
                'name_2': name2,
                'similarity': (name_score + domain_score)/2
            })
            # print(rows.loc[i, ['company_name_norm', 'website_domain_norm', 'main_country']])
            # print(rows.loc[j, ['company_name_norm', 'website_domain_norm', 'main_country']])

duplicates_df = pd.DataFrame(duplicate_pairs)

print("Likely duplicate company pairs found:")
print(duplicates_df.head(10))

# TF-IDF -> 2d visualization
pca = PCA(n_components=2, random_state=42)
reduced = pca.fit_transform(X.toarray())

#-scatterplot
# scatterPlot(df, reduced)

# map each index to a group ID
group_map = {}
current_group_id = 0

for _, row in duplicates_df.iterrows():
    idx1 = row['index_1']
    idx2 = row['index_2']

    group1 = group_map.get(idx1)
    group2 = group_map.get(idx2)

    if group1 is not None and group2 is not None:
        for key, val in group_map.items():
            if val == group2:
                group_map[key] = group1
    elif group1 is not None:
        group_map[idx2] = group1
    elif group2 is not None:
        group_map[idx1] = group2
    else:
        group_map[idx1] = current_group_id
        group_map[idx2] = current_group_id
        current_group_id += 1

# invert group map
groups = defaultdict(list)
for idx, group_id in group_map.items():
    groups[group_id].append(idx)

# consolidate
def consolidate_group(group, df):
    group_df = df.loc[group]
    combined = {}

    for col in df.columns:
        values = group_df[col].dropna().unique()

        if df[col].dtype == 'object':
            combined[col] = ' | '.join(sorted(set([v for v in values if v.strip()])))
        else:
            combined[col] = group_df[col].dropna().iloc[0] if not group_df[col].dropna().empty else None

    return combined

consolidated_entries = [consolidate_group(group, df) for group in groups.values()]
grouped_indices = set(idx for group in groups.values() for idx in group)

remaining_df = df.drop(index=list(grouped_indices))
consolidated_df = pd.DataFrame(consolidated_entries)

final_df = pd.concat([consolidated_df, remaining_df], ignore_index=True)
final_df.to_parquet("deduplicated_companies.parquet", index=False)

print(f"Final dataset created: {final_df.shape[0]} rows (from original {df.shape[0]})")
