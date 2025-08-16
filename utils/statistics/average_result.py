from pathlib import Path

import pandas as pd

meta = [
    {
        'name': 'sota',
        'path': Path(r'E:/Code/SparrKULee/SparrKULee/match_mismatch/output/sota'),
        'pattern': 'p-10_nc-5_seed-*/test_results.csv'
    },
    {
        'name': 'clip easy',
        'path': Path(r'E:/Code/SparrKULee/SparrKULee/match_mismatch/output/clip_cls'),
        'pattern': 'p-10_nc-5_bsz-32_easy_seed-*_PEFT/test_results.csv'
    },
    {
        'name': 'clip difficult',
        'path': Path(r'E:/Code/SparrKULee/SparrKULee/match_mismatch/output/clip_cls'),
        'pattern': 'p-10_nc-5_bsz-32_difficult_seed-*_PEFT/test_results.csv'
    }
]
index = 2

dir_path = meta[index]['path']
pattern = meta[index]['pattern']
columns = ['test_acc', 'test_loss', 'test_total', 'test_correct']
test_results = pd.DataFrame(
    columns=columns,
)
count = 0
for file in dir_path.glob(pattern):
    df = pd.read_csv(
        file,
        index_col='Subject',
        converters={
            'test_acc': lambda x: float(x.strip('%')) / 100,
        }
    )
    test_results.loc[test_results.shape[0]] = df.loc['Average', columns]
    count += 1

avg_results = test_results.mean()
print(f'Result of {meta[index]["name"]}')
print(f'Averaged over {count} runs.')
print(avg_results)
