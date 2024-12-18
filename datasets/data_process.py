import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

def class_mapping(row):
    mappings = {'障害':0, 'G1': 10, 'G2': 9, 'G3': 8, '(L)': 7, 'オープン': 7,'OP': 7, '3勝': 6, '1600': 6, '2勝': 5, '1000': 5, '1勝': 4, '500': 4, '新馬': 3, '未勝利': 1}
    for key, value in mappings.items():
        if key in row:
            return value
    return 0  # If no mapping is found, return 0
# データの読み込み
yearStart = 2024
yearEnd = 2024
yearList = np.arange(yearStart, yearEnd + 1, 1, int)
df = []
print("ファイル取得：開始")
dirname = os.path.dirname(__file__)
for for_year in yearList:
    var_path = os.path.join(dirname, 'raw_data/'+str(for_year)+'.csv')
    var_data = pd.read_csv(
        var_path,
        encoding="SHIFT-JIS",
        header=0,
        parse_dates=['日付'],
        date_parser=lambda x: pd.to_datetime(x, format='%Y年%m月%d日')
    )
    # '着順'カラムの値を数値に変換しようとして、エラーが発生する場合はNaNにする
    var_data['着順'] = pd.to_numeric(var_data['着順'], errors='coerce')
    # NaNの行を削除する
    var_data = var_data.dropna(subset=['着順'])
    # 必要であれば、'着順'カラムのデータ型を整数に変換する
    var_data['着順'] = var_data['着順'].astype(int)

    # "芝・ダート"が"芝"だけの行を選択
    df.append(var_data[var_data['芝・ダート'] == '芝'])
print("ファイル取得：完了")
print("データ変換：開始")
# DataFrameの結合
df_combined = pd.concat(df, ignore_index=True)

# 既存のコード：走破時間を秒に変換
time_parts = df_combined['走破時間'].str.split(':', expand=True)
seconds = time_parts[0].astype(float) * 60 + time_parts[1].str.split('.', expand=True)[0].astype(float) + time_parts[1].str.split('.', expand=True)[1].astype(float) / 10
# 前方補完
seconds = seconds.fillna(method='ffill')

# 平均と標準偏差を計算
mean_seconds = seconds.mean()
std_seconds = seconds.std()

# 標準化を行う
df_combined['走破時間'] = -((seconds - mean_seconds) / std_seconds)

# 外れ値の処理：-3より小さい値は-3に、2.5より大きい値は2に変換
df_combined['走破時間'] = df_combined['走破時間'].apply(lambda x: -3 if x < -3 else (2 if x > 2.5 else x))

# 2回目の標準化の前に再度平均と標準偏差を計算
mean_seconds_2 = df_combined['走破時間'].mean()
std_seconds_2 = df_combined['走破時間'].std()

# 2回目の標準化
df_combined['走破時間'] = (df_combined['走破時間'] - mean_seconds_2) / std_seconds_2
print('1回目平均' + str(mean_seconds))
print('2回目平均' + str(mean_seconds_2))
print('1回目標準偏差' + str(std_seconds))
print('2回目標準偏差' + str(std_seconds_2))

# データを格納するDataFrameを作成
time_df = pd.DataFrame({
    'Mean': [mean_seconds, mean_seconds_2],
    'Standard Deviation': [std_seconds, std_seconds_2]
})
# indexに名前を付ける
time_df.index = ['First Time', 'Second Time']
# DataFrameをCSVファイルとして出力
#time_df.to_csv(os.path.join(dirname,'timedata/standard_deviation.csv'))

#通過順の平均を出す
pas = df_combined['通過順'].str.split('-', expand=True)
df_combined['通過順'] = pas.astype(float).mean(axis=1)

# mapを使ったラベルの変換
df_combined['クラス'] = df_combined['クラス'].apply(class_mapping)
sex_mapping = {'牡':0, '牝': 1, 'セ': 2}
df_combined['性'] = df_combined['性'].map(sex_mapping)
shiba_mapping = {'芝': 0, 'ダ': 1, '障': 2}
df_combined['芝・ダート'] = df_combined['芝・ダート'].map(shiba_mapping)
mawari_mapping = {'右': 0, '左': 1, '芝': 2, '直': 2}
df_combined['回り'] = df_combined['回り'].map(mawari_mapping)
baba_mapping = {'良': 0, '稍': 1, '重': 2, '不': 3}
df_combined['馬場'] = df_combined['馬場'].map(baba_mapping)
tenki_mapping = {'晴': 0, '曇': 1, '小': 2, '雨': 3, '雪': 4}
df_combined['天気'] = df_combined['天気'].map(tenki_mapping)
print("データ変換：完了")
print("近5走取得：開始")
# '馬'と'日付'に基づいて降順にソート
df_combined.sort_values(by=['馬', '日付'], ascending=[True, False], inplace=True)

features = ['馬番', '騎手', '斤量', 'オッズ', '体重', '体重変化', '上がり', '通過順', '着順', '距離', 'クラス', '走破時間', '芝・ダート', '天気','馬場']
#斤量、周り
# 同じ馬の過去5レースの情報を新しいレース結果にマージ
for i in range(1, 6):
    df_combined[f'日付{i}'] = df_combined.groupby('馬')['日付'].shift(-i)
    for feature in features:
        df_combined[f'{feature}{i}'] = df_combined.groupby('馬')[feature].shift(-i)

# 同じ馬のデータで欠損値を補完
for feature in features:
    for i in range(1, 6):
        df_combined[f'{feature}{i}'] = df_combined.groupby('馬')[f'{feature}{i}'].fillna(method='ffill')

# race_id と 馬 でグルーピングし、各特徴量の最新の値を取得
df_combined = df_combined.groupby(['race_id', '馬'], as_index=False).last()

# race_idでソート
df_combined.sort_values(by='race_id', ascending=False, inplace=True)

print("近5走取得：終了")
# '---' をNaNに置き換える
df_combined.replace('---', np.nan, inplace=True)
print("日付変換：開始")
#距離差と日付差を計算
df_combined = df_combined.assign(
    距離差 = df_combined['距離'] - df_combined['距離1'],
    日付差 = (df_combined['日付'] - df_combined['日付1']).dt.days,
    距離差1 = df_combined['距離1'] - df_combined['距離2'],
    日付差1 = (df_combined['日付1'] - df_combined['日付2']).dt.days,
    距離差2 = df_combined['距離2'] - df_combined['距離3'],
    日付差2 = (df_combined['日付2'] - df_combined['日付3']).dt.days,
    距離差3 = df_combined['距離3'] - df_combined['距離4'],
    日付差3 = (df_combined['日付3'] - df_combined['日付4']).dt.days,
    距離差4 = df_combined['距離4'] - df_combined['距離5'],
    日付差4 = (df_combined['日付4'] - df_combined['日付5']).dt.days
)

# 斤量に関連する列を数値に変換し、変換できないデータはNaNにします。
kinryo_columns = ['斤量', '斤量1', '斤量2', '斤量3', '斤量4','斤量5']
for col in kinryo_columns:
    df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')

# 平均斤量を計算します。
df_combined['平均斤量'] = df_combined[kinryo_columns].mean(axis=1)

# 騎手の勝率
jockey_win_rate = df_combined.groupby('騎手')['着順'].apply(lambda x: (x==1).sum() / x.count()).reset_index()
jockey_win_rate.columns = ['騎手', '騎手の勝率']
#jockey_win_rate.to_csv(os.path.join(dirname,'calc_rate/jockey_win_rate.csv'), index=False)
# '騎手'をキーにしてdf_combinedとjockey_win_rateをマージする
df_combined = pd.merge(df_combined, jockey_win_rate, on='騎手', how='left')

#日付
# 日付カラムから年、月、日を抽出
df_combined['year'] = df_combined['日付'].dt.year
df_combined['month'] = df_combined['日付'].dt.month
df_combined['day'] = df_combined['日付'].dt.day
# (年-yearStart)*365 + 月*30 + 日 を計算し新たな '日付'カラムを作成
df_combined['日付'] = (df_combined['year'] - yearStart) * 365 + df_combined['month'] * 30 + df_combined['day']

df_combined['year'] = df_combined['日付1'].dt.year
df_combined['month'] = df_combined['日付1'].dt.month
df_combined['day'] = df_combined['日付1'].dt.day
df_combined['日付1'] = (df_combined['year'] - yearStart) * 365 + df_combined['month'] * 30 + df_combined['day']

df_combined['year'] = df_combined['日付2'].dt.year
df_combined['month'] = df_combined['日付2'].dt.month
df_combined['day'] = df_combined['日付2'].dt.day
df_combined['日付2'] = (df_combined['year'] - yearStart) * 365 + df_combined['month'] * 30 + df_combined['day']

df_combined['year'] = df_combined['日付3'].dt.year
df_combined['month'] = df_combined['日付3'].dt.month
df_combined['day'] = df_combined['日付3'].dt.day
df_combined['日付3'] = (df_combined['year'] - yearStart) * 365 + df_combined['month'] * 30 + df_combined['day']

df_combined['year'] = df_combined['日付4'].dt.year
df_combined['month'] = df_combined['日付4'].dt.month
df_combined['day'] = df_combined['日付4'].dt.day
df_combined['日付4'] = (df_combined['year'] - yearStart) * 365 + df_combined['month'] * 30 + df_combined['day']

df_combined['year'] = df_combined['日付5'].dt.year
df_combined['month'] = df_combined['日付5'].dt.month
df_combined['day'] = df_combined['日付5'].dt.day
df_combined['日付5'] = (df_combined['year'] - yearStart) * 365 + df_combined['month'] * 30 + df_combined['day']

# 不要となった 'year', 'month', 'day' カラムを削除
df_combined.drop(['year', 'month', 'day'], axis=1, inplace=True)
print("日付変換：終了")

categorical_features = ['馬', '騎手', 'レース名', '開催', '場名', '騎手1', '騎手2', '騎手3', '騎手4', '騎手5']  # カテゴリカル変数の列名を指定してください

# ラベルエンコーディング
for i, feature in enumerate(categorical_features):
    print(f"\rProcessing feature {i+1}/{len(categorical_features)}", end="")
    le = LabelEncoder()
    df_combined[feature] = le.fit_transform(df_combined[feature])

# エンコーディングとスケーリング後のデータを確認
print("ファイル出力：開始")
df_combined.to_csv(os.path.join(dirname, 'encoded/encoded_testdata.csv'), index=False)
print("ファイル出力：終了")