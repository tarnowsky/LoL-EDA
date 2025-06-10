# -*- coding: utf-8 -*-
"""
# Eksploracyjna analiza danych z gier League of Legends
# Data: 20.05.2025
# Cel: Analiza statystyk z pierwszych 10 minut gier League of Legends rangi Diamond I-Master 
# i zbadanie czynników wpływających na zwycięstwo niebieskiej drużyny
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import missingno as msno
from sklearn.preprocessing import StandardScaler
import os

# Ustawienie stylu wizualizacji
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# -----------------------------------------
# CZĘŚĆ 1: WCZYTANIE I OPIS DANYCH
# -----------------------------------------

# Wczytanie danych
print("Wczytywanie danych...")
df = pd.read_csv("high_diamond_ranked_10min.csv")

# Wyświetlenie podstawowych informacji o zbiorze danych
print("\n--- INFORMACJE O ZBIORZE DANYCH ---")
print(f"Liczba wierszy (gier): {df.shape[0]}")
print(f"Liczba kolumn (atrybutów): {df.shape[1]}")
print("\nPierwsze 5 wierszy danych:")
print(df.head())

# Informacje o typach danych i wartościach brakujących
print("\nInformacje o typach danych i wartościach brakujących:")
print(df.info())

# Statystyki opisowe dla wszystkich atrybutów numerycznych
print("\nStatystyki opisowe:")
print(df.describe())

# -----------------------------------------
# CZĘŚĆ 2: EKSPLORACYJNA ANALIZA DANYCH (EDA)
# -----------------------------------------


# Sprawdzenie wartości brakujących
print("\n--- SPRAWDZENIE WARTOŚCI BRAKUJĄCYCH ---")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# Wizualizacja brakujących wartości
plt.figure(figsize=(10, 6))
msno.matrix(df)
plt.title('Wizualizacja brakujących wartości')
plt.tight_layout()
plt.savefig('missing_values.png')

# Sprawdzenie rozkładu zmiennej docelowej (blueWins)
print("\n--- ROZKŁAD ZMIENNEJ DOCELOWEJ ---")
blue_win_counts = df['blueWins'].value_counts()
blue_win_percentage = df['blueWins'].mean() * 100
print(f"Liczba zwycięstw drużyny niebieskiej: {blue_win_counts[1]} ({blue_win_percentage:.2f}%)")
print(f"Liczba przegranych drużyny niebieskiej: {blue_win_counts[0]} ({100-blue_win_percentage:.2f}%)")

plt.figure(figsize=(8, 6))
sns.countplot(x='blueWins', data=df, palette=['red', 'blue'])
plt.title('Rozkład zwycięstw drużyny niebieskiej')
plt.xlabel('Czy drużyna niebieska wygrała?')
plt.xticks([0, 1], ['Nie (0)', 'Tak (1)'])
plt.ylabel('Liczba gier')
plt.tight_layout()
plt.savefig('blue_wins_distribution.png')

# -----------------------------------------
# Analiza rozkładów atrybutów
# -----------------------------------------

# Utworzenie katalogów do zapisu histogramów
os.makedirs("histograms/blue", exist_ok=True)
os.makedirs("histograms/red", exist_ok=True)

# Funkcja do tworzenia histogramów dla wybranych atrybutów
def plot_histograms(dataframe, columns, rows=3, cols=3):
    fig, axes = plt.subplots(rows, cols, figsize=(18, 15))
    axes = axes.flatten()
    
    for i, column in enumerate(columns):
        if i < len(axes):
            sns.histplot(dataframe[column], kde=True, ax=axes[i])
            axes[i].set_title(f'Rozkład: {column}')
            axes[i].set_xlabel(column)
            axes[i].set_ylabel('Częstość')
    
    plt.tight_layout()
    return fig

# Wybrane atrybuty dla drużyny niebieskiej
blue_attributes = ['blueKills', 'blueDeaths', 'blueAssists', 'blueTotalGold', 
                   'blueAvgLevel', 'blueTotalExperience', 'blueTotalMinionsKilled', 
                   'blueGoldDiff', 'blueExperienceDiff']

# Tworzenie i zapisywanie osobnych histogramów dla każdego atrybutu drużyny niebieskiej
for attr in blue_attributes:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[attr], kde=True)
    plt.title(f'Rozkład: {attr} (drużyna niebieska)')
    plt.xlabel(attr)
    plt.ylabel('Częstość')
    plt.tight_layout()
    plt.savefig(f'histograms/blue/blue_{attr}_histogram.png')
    plt.close()

# Wybrane atrybuty dla drużyny czerwonej
red_attributes = ['redKills', 'redDeaths', 'redAssists', 'redTotalGold', 
                  'redAvgLevel', 'redTotalExperience', 'redTotalMinionsKilled', 
                  'redGoldDiff', 'redExperienceDiff']

# Tworzenie i zapisywanie osobnych histogramów dla każdego atrybutu drużyny czerwonej
for attr in red_attributes:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[attr], kde=True)
    plt.title(f'Rozkład: {attr} (drużyna czerwona)')
    plt.xlabel(attr)
    plt.ylabel('Częstość')
    plt.tight_layout()
    plt.savefig(f'histograms/red/red_{attr}_histogram.png')
    plt.close()

# Analiza rozkładu atrybutów różnic (diffs)
diff_attributes = ['blueGoldDiff', 'blueExperienceDiff', 'redGoldDiff', 'redExperienceDiff']
diff_hist_fig = plot_histograms(df, diff_attributes, rows=2, cols=2)
diff_hist_fig.suptitle('Rozkłady atrybutów różnicowych', fontsize=20)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig('diff_histograms.png')

# -----------------------------------------
# Analiza korelacji
# -----------------------------------------

# Wybór istotnych kolumn do analizy korelacji
important_columns = ['blueWins', 'blueKills', 'blueDeaths', 'blueAssists', 
                    'blueEliteMonsters', 'blueDragons', 'blueHeralds',
                    'blueTowersDestroyed', 'blueTotalGold', 'blueAvgLevel',
                    'blueTotalExperience', 'blueTotalMinionsKilled',
                    'blueTotalJungleMinionsKilled', 'blueGoldDiff', 
                    'blueExperienceDiff', 'blueCSPerMin', 'blueGoldPerMin',
                    'redKills', 'redDeaths', 'redAssists', 
                    'redEliteMonsters', 'redDragons', 'redHeralds',
                    'redTowersDestroyed', 'redTotalGold', 'redAvgLevel',
                    'redTotalExperience', 'redTotalMinionsKilled',
                    'redTotalJungleMinionsKilled', 'redGoldDiff', 
                    'redExperienceDiff', 'redCSPerMin', 'redGoldPerMin']

# Obliczanie macierzy korelacji
correlation_matrix = df[important_columns].corr()

# Korelacje ze zmienną docelową (blueWins)
print("\n--- KORELACJE ZE ZMIENNĄ DOCELOWĄ (blueWins) ---")
blue_wins_corr = correlation_matrix['blueWins'].sort_values(ascending=False)
print(blue_wins_corr)

# Wizualizacja korelacji ze zmienną docelową
plt.figure(figsize=(12, 10))
top_correlations = blue_wins_corr[1:11]  # Top 10 bez samej zmiennej blueWins
sns.barplot(x=top_correlations.values, y=top_correlations.index)
plt.title('Top 10 atrybutów najbardziej skorelowanych ze zwycięstwem drużyny niebieskiej')
plt.xlabel('Współczynnik korelacji Pearsona')
plt.tight_layout()
plt.savefig('top_correlations.png')

# Wizualizacja macierzy korelacji (heatmapa)
plt.figure(figsize=(20, 16))
mask = np.triu(correlation_matrix)
sns.heatmap(correlation_matrix, annot=False, mask=mask, cmap='coolwarm', 
            vmin=-1, vmax=1, center=0, linewidths=0.5)
plt.title('Macierz korelacji dla atrybutów gry', fontsize=18)
plt.tight_layout()
plt.savefig('correlation_matrix.png')

# Bardziej szczegółowa analiza najważniejszych korelacji
top_features = ['blueGoldDiff', 'blueExperienceDiff', 'blueTotalGold', 'blueKills', 'blueTowersDestroyed']

for feature in top_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='blueWins', y=feature, data=df)
    plt.title(f'{feature} w zależności od wyniku gry')
    plt.xlabel('Czy drużyna niebieska wygrała?')
    plt.xticks([0, 1], ['Nie (0)', 'Tak (1)'])
    plt.tight_layout()
    plt.savefig(f'boxplot_{feature}.png')

# -----------------------------------------
# Porównanie statystyk dla wygranych i przegranych gier
# -----------------------------------------


# Porównanie średnich wartości dla wygranych i przegranych gier
print("\n--- PORÓWNANIE ŚREDNICH WARTOŚCI DLA WYGRANYCH I PRZEGRANYCH GIER ---")
wins_stats = df[df['blueWins'] == 1].mean()
losses_stats = df[df['blueWins'] == 0].mean()
comparison = pd.DataFrame({'Wygrane (średnia)': wins_stats, 'Przegrane (średnia)': losses_stats})
comparison['Różnica (%)'] = ((wins_stats - losses_stats) / losses_stats * 100).round(2)

# Filtrujemy kolumny, które faktycznie istnieją w ramce danych comparison
valid_columns = [col for col in important_columns[1:] if col in comparison.index]
print(comparison.loc[valid_columns])

# Wizualizacja porównania wybranych atrybutów dla wygranych i przegranych gier
compare_attributes = ['blueKills', 'blueTotalGold', 'blueAvgLevel', 'blueDragons', 'blueTowersDestroyed']

plt.figure(figsize=(14, 8))
df_melted = pd.melt(df, id_vars=['blueWins'], value_vars=compare_attributes)
sns.barplot(x='variable', y='value', hue='blueWins', data=df_melted)
plt.title('Porównanie kluczowych statystyk dla wygranych i przegranych gier')
plt.xlabel('Atrybut')
plt.ylabel('Średnia wartość')
plt.legend(title='Drużyna niebieska wygrała', labels=['Nie', 'Tak'])
plt.tight_layout()
plt.savefig('win_loss_comparison.png')

# -----------------------------------------
# Analiza jakości danych
# -----------------------------------------


# Sprawdzenie punktów odstających
print("\n--- ANALIZA PUNKTÓW ODSTAJĄCYCH ---")
outlier_columns = ['blueKills', 'blueDragons', 'blueHeralds', 'blueTowersDestroyed',
                   'redKills', 'redDragons', 'redHeralds', 'redTowersDestroyed']

outliers_summary = pd.DataFrame(columns=['Atrybut', 'Liczba_outlierów', 'Procent_outlierów'])

for column in outlier_columns:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    outlier_count = len(outliers)
    outlier_percent = (outlier_count / len(df)) * 100
    
    outliers_summary = pd.concat([outliers_summary, pd.DataFrame({
        'Atrybut': [column],
        'Liczba_outlierów': [outlier_count],
        'Procent_outlierów': [round(outlier_percent, 2)]
    })], ignore_index=True)
    
    # Wizualizacja wykrywania punktów odstających za pomocą boxplotów
    if outlier_count > 0:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df[column])
        plt.title(f'Wykrywanie punktów odstających: {column}')
        plt.xlabel(column)
        plt.tight_layout()
        plt.savefig(f'outliers_{column}.png')

print(outliers_summary)

# Sprawdzenie spójności danych
print("\n--- SPRAWDZENIE SPÓJNOŚCI DANYCH ---")

# Sprawdzenie, czy suma zabójstw jednej drużyny zgadza się z liczbą śmierci drugiej drużyny
kills_deaths_check = df['blueKills'].sum() == df['redDeaths'].sum() and df['redKills'].sum() == df['blueDeaths'].sum()
print(f"Czy suma zabójstw = suma śmierci przeciwnej drużyny: {kills_deaths_check}")

# Sprawdzenie, czy wartości GoldDiff są przeciwne dla obu drużyn
gold_diff_check = (df['blueGoldDiff'] + df['redGoldDiff']).abs().mean()
print(f"Średnia bezwzględna różnica pomiędzy blueGoldDiff i redGoldDiff: {gold_diff_check}")

# Sprawdzenie, czy wartości ExperienceDiff są przeciwne dla obu drużyn
exp_diff_check = (df['blueExperienceDiff'] + df['redExperienceDiff']).abs().mean()
print(f"Średnia bezwzględna różnica pomiędzy blueExperienceDiff i redExperienceDiff: {exp_diff_check}")

# Sprawdzenie wartości binarnych (czy zawsze tylko jedna drużyna ma FirstBlood)
first_blood_check = df['blueFirstBlood'] + df['redFirstBlood'] == 1
first_blood_valid = first_blood_check.all()
print(f"Czy dokładnie jedna drużyna ma FirstBlood w każdej grze: {first_blood_valid}")

# Sprawdzenie wartości atrybutów dla 10 minut gry
print("\n--- WERYFIKACJA ATRYBUTÓW CZASOWYCH ---")
cs_per_min_check = (df['blueCSPerMin'] * 10).round(0) == df['blueTotalMinionsKilled'].round(0)
gold_per_min_check = (df['blueGoldPerMin'] * 10).round(0) == df['blueTotalGold'].round(0)

print(f"Procent gier, gdzie blueCSPerMin * 10 ≈ blueTotalMinionsKilled: {cs_per_min_check.mean() * 100:.2f}%")
print(f"Procent gier, gdzie blueGoldPerMin * 10 ≈ blueTotalGold: {gold_per_min_check.mean() * 100:.2f}%")

# -----------------------------------------
# Analiza wpływu czynników na wygraną
# -----------------------------------------

# Analiza znaczenia czynników dla przewidywania wyniku gry
print("\n--- ANALIZA ZNACZENIA CZYNNIKÓW DLA WYNIKU GRY ---")

# Przygotowanie danych dla modelu
X = df.drop(['gameId', 'blueWins'], axis=1)
y = df['blueWins']

# Standaryzacja cech
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Obliczenie t-statystyk dla każdej cechy
t_stats = []
p_values = []
feature_names = X.columns

for i, feature in enumerate(feature_names):
    t_stat, p_val = stats.ttest_ind(
        X_scaled[y == 1][:, i],
        X_scaled[y == 0][:, i],
        equal_var=False
    )
    t_stats.append(np.abs(t_stat))
    p_values.append(p_val)

# Tworzenie DataFrame z wynikami
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'T-statistic': t_stats,
    'P-value': p_values
})

# Sortowanie cech według wartości bezwzględnej t-statystyki
feature_importance = feature_importance.sort_values('T-statistic', ascending=False)
feature_importance['Significance'] = feature_importance['P-value'].apply(
    lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
)

print(feature_importance.head(15))

# Wizualizacja najważniejszych czynników
plt.figure(figsize=(12, 10))
top_features = feature_importance.head(15)
sns.barplot(x='T-statistic', y='Feature', data=top_features)
plt.title('15 najważniejszych czynników wpływających na wynik gry')
plt.xlabel('Wartość bezwzględna statystyki t')
plt.tight_layout()
plt.savefig('feature_importance.png')

# -----------------------------------------
# Podsumowanie
# -----------------------------------------

print("\n--- EKSPLORACJA DANYCH ZAKOŃCZONA ---")
print("Wszystkie wykresy zostały zapisane jako pliki PNG.")
print("Wyniki analizy można wykorzystać do przygotowania raportu i budowy modeli predykcyjnych.")