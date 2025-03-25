import pandas as pd

# 读取train.csv
df = pd.read_csv('E:/test_filter.csv')

# 保留dataset_name列名为inaturalist的行
filtered_df_inat = df[df['dataset_name'] == 'inaturalist']

# # 去重，确保 first_image_id 不重复
filtered_df_inat = filtered_df_inat.drop_duplicates(subset=['dataset_image_ids'])

filtered_df_inat = filtered_df_inat[:1000]

# 保存到test_split.csv
filtered_df_inat.to_csv('E:/test_split_inat_filter.csv', index=False)

# ————————————————————————————————————————————————————————————————————————
# 读取 train.csv
df = pd.read_csv('E:/test_filter.csv')

# 保留dataset_name列名为inaturalist的行
filtered_df_glv2 = df[df['dataset_name'] == 'landmarks']

# # 获取第一个 dataset_image_id
# df['first_image_id'] = df['dataset_image_ids'].astype(str).str.split('|').str[0]

# # 筛选以 00-03 开头的行
# filtered_df = df[df['first_image_id'].str.match(r'^(00|01|02|03|04|05|06|07|08)')]

# # 去重，确保 first_image_id 不重复
# filtered_df = filtered_df.drop_duplicates(subset=['first_image_id'])

# # 取前 1000 行
filtered_df_glv2 = filtered_df_glv2[:1000]

# 保存到 glv2_split.csv
filtered_df_glv2.to_csv('E:/test_split_glv2_filter.csv', index=False)

