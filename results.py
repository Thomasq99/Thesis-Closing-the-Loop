import pandas as pd

df = pd.read_excel('./Questionnaire.xlsx', 'Formulierreacties 1')
df = df.iloc[1:, -4:]
df = df.drop(18)
position = df.iloc[:, 0].apply(lambda x: x.index('E') + 1)
position_changed = df.iloc[:, 1].apply(lambda x: x.index('E') + 1)
print(position.mean(), position_changed.mean())


position = df.iloc[:, 2].apply(lambda x: x.index('E') + 1)
position_changed = df.iloc[:, 3].apply(lambda x: x.index('E') + 1)
print(position.mean(), position_changed.mean())
print(position)

print(position_changed)