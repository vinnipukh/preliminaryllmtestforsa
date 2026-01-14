import pandas as pd
df = pd.read_csv('TRSAv1.csv',encoding='utf-8')
print(df['score'].value_counts())
df_Positive = df[df['score'] == 'Positive'].sample(n=1000, random_state=42)
df_Negative = df[df['score'] == 'Negative'].sample(n=1000, random_state=42)
df_Neutral = df[df['score'] == 'Neutral'].sample(n=1000, random_state=42)
df_final = pd.concat([df_Positive,df_Negative,df_Neutral])
df_final = df_final.sample(frac=1).reset_index(drop=True)
df_final.to_csv('TRSAv1_sample_3k.csv', index=False, encoding='utf-8-sig')