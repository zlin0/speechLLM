import pandas as pd

<<<<<<< HEAD
def save_csv(df, dataset, set, target="/export/fs05/lzhan268/workspace/03llm/speechLLM/data/"):
    columns = ['transcript','gender','emotion','age','accent', 'noises', 'summary']
    df['dataset'] = dataset
    df['set'] = set
    df['isspeech'] = True
    assert 'audio_path' in df.columns, f"audio_path not found in {dataset} {set}"
    assert 'audio_len' in df.columns, f"audio_len not found in {dataset} {set}"
    for col in columns:
        if col not in df.columns: df[col]=None

    df.to_csv(f"{target}{dataset}_{set}.csv", index=False)
    print(f"{dataset} {set} saved! total length: {df.shape}")
