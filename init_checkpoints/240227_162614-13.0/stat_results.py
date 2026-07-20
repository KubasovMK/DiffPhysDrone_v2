import json
import sys
import pandas as pd


for filename in sys.argv[1:]:
    data = []
    with open(filename) as f:
        f.readline()
        for line in f:
            data.append(json.loads(line))
    if data:
        df = pd.DataFrame(data)
        df['avg_speed'] = df['travelled_dist'] / df ['travelled_time']
        print(filename, df[df['success']]['avg_speed'].mean(), df['success'].mean())
