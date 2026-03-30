import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path

base = Path(__file__).resolve().parents[1]
(base/'data'/'raw').mkdir(parents=True, exist_ok=True)
(base/'data'/'clean').mkdir(parents=True, exist_ok=True)
(base/'visualizations').mkdir(parents=True, exist_ok=True)

df = px.data.gapminder().copy()
df.to_csv(base/'data'/'raw'/'gapminder_raw.csv', index=False)

clean = df.copy()
clean['country'] = clean['country'].str.strip()
clean['continent'] = clean['continent'].str.strip()
clean['year'] = clean['year'].astype(int)
clean['lifeExp'] = clean['lifeExp'].astype(float).round(2)
clean['pop'] = clean['pop'].astype(int)
clean['gdpPercap'] = clean['gdpPercap'].astype(float).round(2)
clean['gdp_total_usd'] = (clean['gdpPercap'] * clean['pop']).round(0).astype('int64')
clean = clean.rename(columns={
    'lifeExp':'life_expectancy',
    'pop':'population',
    'gdpPercap':'gdp_per_capita'
})
clean.to_csv(base/'data'/'clean'/'gapminder_clean.csv', index=False)

latest = clean[clean['year']==2007].copy()

plt.figure(figsize=(10,6))
for cont, grp in latest.groupby('continent'):
    plt.scatter(grp['gdp_per_capita'], grp['life_expectancy'],
                s=np.clip(grp['population']/2e6, 15, 300), alpha=0.7, label=cont)
plt.xscale('log')
plt.xlabel('GDP per capita, USD (log scale)')
plt.ylabel('Life expectancy, years')
plt.title('Life expectancy and GDP per capita by country, 2007')
plt.legend(title='Continent', frameon=False)
plt.tight_layout()
plt.savefig(base/'visualizations'/'01_lifeexp_vs_gdp_2007.png', dpi=200)
plt.close()

cont_time = clean.groupby(['year','continent']).apply(
    lambda g: np.average(g['life_expectancy'], weights=g['population'])
).reset_index(name='weighted_life_expectancy')

plt.figure(figsize=(10,6))
for cont, grp in cont_time.groupby('continent'):
    plt.plot(grp['year'], grp['weighted_life_expectancy'], marker='o', label=cont)
plt.xlabel('Year')
plt.ylabel('Population-weighted life expectancy, years')
plt.title('Growth of life expectancy by continent, 1952–2007')
plt.legend(title='Continent', frameon=False)
plt.tight_layout()
plt.savefig(base/'visualizations'/'02_lifeexp_by_continent_1952_2007.png', dpi=200)
plt.close()
