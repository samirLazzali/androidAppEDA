
# Projet MAD 
DE SOUSA Emilio & LAZZALI Samir

### Sommaire
- I - INTRODUCTION
- II - DESCRIPTION
 - Inportation et forme du DataSet 
 - Nettoyage des données
- III - STATISTIQUE DESCRIPTIVE UNIDIMENSIONNEL
  - Rating  
  - Category
  - Reviews  
  - Size
  - Installs
  - Type
  - Type (gratuit payant)
  - Price
  - Content Rating
  - Etude de Genres
- IV - STATISTIQUE DESCRIPTIVE BIDIMENSIONNEL (Par rapport au Rating)
  - Category
  - Reviews
  - Size
  - Installs
  - Price
  - Content Rating
  - Genres
- V - STATISTIQUE DESCRIPTIVE MULTIDIMENSIONNELLE
  - ANALYSE EN COMPOSANTE PRINCIPALES (ACP)	
- VIII - CONCLUSION






# INTRODUCTION
#### Dire des choses ici 



```python
# importation des librairies
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph.
from scipy.special import boxcox1p
```

# DESCRIPTION
#### Dire des choeses ici 

## Importation et forme du DataSet


```python
data = pd.read_csv("googleplaystore.csv")
dataType = data.dtypes
print(data.shape)
data.head()
```

    (10841, 13)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>App</th>
      <th>Category</th>
      <th>Rating</th>
      <th>Reviews</th>
      <th>Size</th>
      <th>Installs</th>
      <th>Type</th>
      <th>Price</th>
      <th>Content Rating</th>
      <th>Genres</th>
      <th>Last Updated</th>
      <th>Current Ver</th>
      <th>Android Ver</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Photo Editor &amp; Candy Camera &amp; Grid &amp; ScrapBook</td>
      <td>ART_AND_DESIGN</td>
      <td>4.1</td>
      <td>159</td>
      <td>19M</td>
      <td>10,000+</td>
      <td>Free</td>
      <td>0</td>
      <td>Everyone</td>
      <td>Art &amp; Design</td>
      <td>January 7, 2018</td>
      <td>1.0.0</td>
      <td>4.0.3 and up</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Coloring book moana</td>
      <td>ART_AND_DESIGN</td>
      <td>3.9</td>
      <td>967</td>
      <td>14M</td>
      <td>500,000+</td>
      <td>Free</td>
      <td>0</td>
      <td>Everyone</td>
      <td>Art &amp; Design;Pretend Play</td>
      <td>January 15, 2018</td>
      <td>2.0.0</td>
      <td>4.0.3 and up</td>
    </tr>
    <tr>
      <th>2</th>
      <td>U Launcher Lite – FREE Live Cool Themes, Hide ...</td>
      <td>ART_AND_DESIGN</td>
      <td>4.7</td>
      <td>87510</td>
      <td>8.7M</td>
      <td>5,000,000+</td>
      <td>Free</td>
      <td>0</td>
      <td>Everyone</td>
      <td>Art &amp; Design</td>
      <td>August 1, 2018</td>
      <td>1.2.4</td>
      <td>4.0.3 and up</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sketch - Draw &amp; Paint</td>
      <td>ART_AND_DESIGN</td>
      <td>4.5</td>
      <td>215644</td>
      <td>25M</td>
      <td>50,000,000+</td>
      <td>Free</td>
      <td>0</td>
      <td>Teen</td>
      <td>Art &amp; Design</td>
      <td>June 8, 2018</td>
      <td>Varies with device</td>
      <td>4.2 and up</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Pixel Draw - Number Art Coloring Book</td>
      <td>ART_AND_DESIGN</td>
      <td>4.3</td>
      <td>967</td>
      <td>2.8M</td>
      <td>100,000+</td>
      <td>Free</td>
      <td>0</td>
      <td>Everyone</td>
      <td>Art &amp; Design;Creativity</td>
      <td>June 20, 2018</td>
      <td>1.1</td>
      <td>4.4 and up</td>
    </tr>
  </tbody>
</table>
</div>



#### On calcule le pourcentage de valeur null en fonction des variables


```python
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(6)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total</th>
      <th>Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Rating</th>
      <td>1474</td>
      <td>0.135965</td>
    </tr>
    <tr>
      <th>Current Ver</th>
      <td>8</td>
      <td>0.000738</td>
    </tr>
    <tr>
      <th>Android Ver</th>
      <td>3</td>
      <td>0.000277</td>
    </tr>
    <tr>
      <th>Content Rating</th>
      <td>1</td>
      <td>0.000092</td>
    </tr>
    <tr>
      <th>Type</th>
      <td>1</td>
      <td>0.000092</td>
    </tr>
    <tr>
      <th>Last Updated</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



## Nettoyage des données 



```python

data.dropna(how ='any', inplace = True)
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(6)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total</th>
      <th>Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Android Ver</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Current Ver</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Last Updated</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Genres</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Content Rating</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Price</th>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#forme actuelle :
print(data.shape)

```

    (9360, 13)


# STATISTIQUE DESCRIPTIVE UNIDIMENSIONNEL

## Rating


```python
data.Rating.describe()
```




    count    9360.000000
    mean        4.191838
    std         0.515263
    min         1.000000
    25%         4.000000
    50%         4.300000
    75%         4.500000
    max         5.000000
    Name: Rating, dtype: float64




```python
g = sns.kdeplot(data.Rating, color="Red", shade = True)
g.set_xlabel("Notation")
g.set_ylabel("Fréquence")
plt.title('Distribution des notes',size = 20)
#plt.savefig('rating_freq.png')
```




    Text(0.5, 1.0, 'Distribution des notes')




![png](output_14_1.png)


## Category


```python
print( len(data['Category'].unique()) , "categories")

print("\n", data['Category'].unique())
```

    33 categories
    
     ['ART_AND_DESIGN' 'AUTO_AND_VEHICLES' 'BEAUTY' 'BOOKS_AND_REFERENCE'
     'BUSINESS' 'COMICS' 'COMMUNICATION' 'DATING' 'EDUCATION' 'ENTERTAINMENT'
     'EVENTS' 'FINANCE' 'FOOD_AND_DRINK' 'HEALTH_AND_FITNESS' 'HOUSE_AND_HOME'
     'LIBRARIES_AND_DEMO' 'LIFESTYLE' 'GAME' 'FAMILY' 'MEDICAL' 'SOCIAL'
     'SHOPPING' 'PHOTOGRAPHY' 'SPORTS' 'TRAVEL_AND_LOCAL' 'TOOLS'
     'PERSONALIZATION' 'PRODUCTIVITY' 'PARENTING' 'WEATHER' 'VIDEO_PLAYERS'
     'NEWS_AND_MAGAZINES' 'MAPS_AND_NAVIGATION']



```python
g = sns.countplot(x="Category",data=data, palette = "Set1")
g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="right")
g 
plt.title('Nombre d\'application pour chaque catégorie' ,size = 20)
#plt.savefig('count_cat.png',bbox_inches='tight')
```




    Text(0.5, 1.0, "Nombre d'application pour chaque catégorie")




![png](output_17_1.png)


#### Les catégories Jeu et Famille sont les plus populaires pour les applications.

## Reviews


```python
data['Reviews'].head()

```




    0       159
    1       967
    2     87510
    3    215644
    4       967
    Name: Reviews, dtype: object



#### Les données sont encore dans le type d'objet, nous avons besoin de convertir les en int


```python
data['Reviews'] = data['Reviews'].apply(lambda x: int(x))
```


```python
g = sns.kdeplot(data.Reviews, color="Green", shade = True)
g.set_xlabel("Reviews")
g.set_ylabel("Frequence")
plt.title('Distribution des Reveiw',size = 20)
```




    Text(0.5, 1.0, 'Distribution des Reveiw')




![png](output_23_1.png)


#### La plupart des applications ont moins d'un million d'évaluations. Évidemment, les applications bien connues ont beaucoup d'évaluations

## Size 


```python
data['Size'].unique()[:30]
```




    array(['19M', '14M', '8.7M', '25M', '2.8M', '5.6M', '29M', '33M', '3.1M',
           '28M', '12M', '20M', '21M', '37M', '5.5M', '17M', '39M', '31M',
           '4.2M', '23M', '6.0M', '6.1M', '4.6M', '9.2M', '5.2M', '11M',
           '24M', 'Varies with device', '9.4M', '15M'], dtype=object)



#### Les données sont toujours dans du type objet et contiennent l'unité, il y aussi et des **Varies with device** à supprimer


```python
len(data[data.Size == 'Varies with device'])

```




    1637



#### **Nettoyage des donées** : On les convertis d'abord en NA.
On supprime les unité **k** ou **M** 


```python
data['Size'].replace('Varies with device', np.nan, inplace = True )

```

# Et la pourquoi tu fillna(1) pour remplacer les na et apres tu veux remplacer les na par mean ???



```python
data.Size = (data.Size.replace(r'[kM]+$', '', regex=True).astype(float) * \
             data.Size.str.extract(r'[\d\.]+([KM]+)', expand=False)
            .fillna(1)
            .replace(['k','M'], [10**3, 10**6]).astype(int))
```

# La tu affecte la mean à tes NA mais tu les a replacé par 1 en haut 


```python
data['Size'].fillna(data.groupby('Category')['Size'].transform('mean'),inplace = True)
```


```python
g = sns.kdeplot(data.Size, color="Red", shade = True)
g.set_xlabel("Taille")
g.set_ylabel("Fréquence")
plt.title('Distribution des tailles',size = 20)
```




    Text(0.5, 1.0, 'Distribution des tailles')




![png](output_35_1.png)


## Installs


```python
data['Installs'].unique()[:10]

```




    array(['10,000+', '500,000+', '5,000,000+', '50,000,000+', '100,000+',
           '50,000+', '1,000,000+', '10,000,000+', '5,000+', '100,000,000+'],
          dtype=object)



#### Les données sont toujours dans le type d'objet et contiennent le signe +.
**Nettoyage des donées**, on les transforme  
- 0 = 1+
- 1 = 5+
- 2 = 10+
- etc


```python
data.Installs = data.Installs.apply(lambda x: x.replace(',',''))
data.Installs = data.Installs.apply(lambda x: x.replace('+',''))
data.Installs = data.Installs.apply(lambda x: int(x))
data['Installs'].unique()


```




    array([     10000,     500000,    5000000,   50000000,     100000,
                50000,    1000000,   10000000,       5000,  100000000,
           1000000000,       1000,  500000000,        100,        500,
                   10,          5,         50,          1])




```python
Sorted_value = sorted(list(data['Installs'].unique()))
data['Installs'].replace(Sorted_value,range(0,len(Sorted_value),1), inplace = True )
data['Installs'].head()
```




    0     8
    1    11
    2    13
    3    15
    4    10
    Name: Installs, dtype: int64



# J'ai pas compris cette ligne au dessus : data['Installs'].replace(Sorted_value,range(0,len(Sorted_value),1), inplace = True ) ???




```python
g = sns.kdeplot(data.Installs, color="Red", shade = True)
g.set_xlabel("Installation")
g.set_ylabel("Fréquence")
plt.title('Distribution des installations',size = 20)
```




    Text(0.5, 1.0, 'Distribution des installations')




![png](output_42_1.png)


##  Type (gratuit payant)


```python
labels =data['Type'].value_counts(sort = True).index
sizes = data['Type'].value_counts(sort = True)


colors = ["palegreen","orangered"]
explode = (0.1,0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=270,)

plt.title('Pourcentage d\'application gratuite',size = 20)
plt.show()
```


![png](output_44_0.png)


#### La plupart des applications sont gratuites (93,1%).

## Price


```python
data.Price.unique()[:30]
```




    array(['0', '$4.99', '$3.99', '$6.99', '$7.99', '$5.99', '$2.99', '$3.49',
           '$1.99', '$9.99', '$7.49', '$0.99', '$9.00', '$5.49', '$10.00',
           '$24.99', '$11.99', '$79.99', '$16.99', '$14.99', '$29.99',
           '$12.99', '$2.49', '$10.99', '$1.50', '$19.99', '$15.99', '$33.99',
           '$39.99', '$3.95'], dtype=object)



#### **Nettoyage des donées**, on supprime le $ des prix avant de les convertir en float


```python

data.Price = data.Price.apply(lambda x: x.replace('$',''))
data['Price'] = data['Price'].apply(lambda x: float(x))

```


```python
data['Price'].describe()

```




    count    9360.000000
    mean        0.961279
    std        15.821640
    min         0.000000
    25%         0.000000
    50%         0.000000
    75%         0.000000
    max       400.000000
    Name: Price, dtype: float64



#### Le prix moyen est d'environ 0,96, mais la plupart sont gratuits (8715/9360).



```python
print( 'L\'application la plus chère est à 400 dollars : ')
data[data['Price'] == 400]['App']

```

    L'application la plus chère est à 400 dollars : 





    4367    I'm Rich - Trump Edition
    Name: App, dtype: object




```python
g = sns.kdeplot(data.Price, color="Red", shade = True)
g.set_xlabel("Prix")
g.set_ylabel("Fréquence")
plt.title('Distribution du prix',size = 20)
```




    Text(0.5, 1.0, 'Distribution du prix')




![png](output_53_1.png)


## Content Rating


```python
data['Content Rating'].unique()

```




    array(['Everyone', 'Teen', 'Everyone 10+', 'Mature 17+',
           'Adults only 18+', 'Unrated'], dtype=object)




```python
# Comme il n'y a qu'une line avec
# un ContenRating : Unrated on le supprime du dataset 
data = data[data['Content Rating'] != 'Unrated']
```

#### Nous reviendrons sur cette variable lors de l'analyse 2D

## Etude de Genres


```python
print( len(data['Genres'].unique()) , "genres")

print("\n", data['Genres'].unique())
```

    115 genres
    
     ['Art & Design' 'Art & Design;Pretend Play' 'Art & Design;Creativity'
     'Auto & Vehicles' 'Beauty' 'Books & Reference' 'Business' 'Comics'
     'Comics;Creativity' 'Communication' 'Dating' 'Education;Education'
     'Education' 'Education;Creativity' 'Education;Music & Video'
     'Education;Action & Adventure' 'Education;Pretend Play'
     'Education;Brain Games' 'Entertainment' 'Entertainment;Music & Video'
     'Entertainment;Brain Games' 'Entertainment;Creativity' 'Events' 'Finance'
     'Food & Drink' 'Health & Fitness' 'House & Home' 'Libraries & Demo'
     'Lifestyle' 'Lifestyle;Pretend Play' 'Adventure;Action & Adventure'
     'Arcade' 'Casual' 'Card' 'Casual;Pretend Play' 'Action' 'Strategy'
     'Puzzle' 'Sports' 'Music' 'Word' 'Racing' 'Casual;Creativity'
     'Casual;Action & Adventure' 'Simulation' 'Adventure' 'Board' 'Trivia'
     'Role Playing' 'Simulation;Education' 'Action;Action & Adventure'
     'Casual;Brain Games' 'Simulation;Action & Adventure'
     'Educational;Creativity' 'Puzzle;Brain Games' 'Educational;Education'
     'Card;Brain Games' 'Educational;Brain Games' 'Educational;Pretend Play'
     'Entertainment;Education' 'Casual;Education' 'Music;Music & Video'
     'Racing;Action & Adventure' 'Arcade;Pretend Play'
     'Role Playing;Action & Adventure' 'Simulation;Pretend Play'
     'Puzzle;Creativity' 'Sports;Action & Adventure'
     'Educational;Action & Adventure' 'Arcade;Action & Adventure'
     'Entertainment;Action & Adventure' 'Puzzle;Action & Adventure'
     'Strategy;Action & Adventure' 'Music & Audio;Music & Video'
     'Health & Fitness;Education' 'Adventure;Education' 'Board;Brain Games'
     'Board;Action & Adventure' 'Board;Pretend Play' 'Casual;Music & Video'
     'Role Playing;Pretend Play' 'Entertainment;Pretend Play'
     'Video Players & Editors;Creativity' 'Card;Action & Adventure' 'Medical'
     'Social' 'Shopping' 'Photography' 'Travel & Local'
     'Travel & Local;Action & Adventure' 'Tools' 'Tools;Education'
     'Personalization' 'Productivity' 'Parenting' 'Parenting;Music & Video'
     'Parenting;Brain Games' 'Parenting;Education' 'Weather'
     'Video Players & Editors' 'Video Players & Editors;Music & Video'
     'News & Magazines' 'Maps & Navigation'
     'Health & Fitness;Action & Adventure' 'Educational' 'Casino'
     'Adventure;Brain Games' 'Lifestyle;Education'
     'Books & Reference;Education' 'Puzzle;Education'
     'Role Playing;Brain Games' 'Strategy;Education' 'Racing;Pretend Play'
     'Communication;Creativity' 'Strategy;Creativity']


#### Les genres sont varier car ils sont parfois ratacher a un sous genre : `Arcade;Action & Adventure`

**Nettoyage des données**

#### On va donc ne conserver que le premier Genres et les grouper pour connaitre la répartition dans les Genres principaux. On va aussi fusionner `Group Music & Audio  et  Music`
#### il en reste 47 contre les 115 du debut.


```python
data['Genres'] = data['Genres'].str.split(';').str[0]
data['Genres'].replace('Music & Audio', 'Music',inplace = True)
print( len(data['Genres'].unique()) , "genres")


```

    47 genres


## Etude de LastUpdated

#### Last Update est toujours au format String, nous avons besoin de la transformer la tracer

# STATISTIQUE DESCRIPTIVE BIDIMENSIONNEL (par rapport au Rating)

## Rating / Category


```python
g = sns.catplot(x="Category",y="Rating",data=data, kind="box", height = 10 ,
palette = "Set1")
g.despine(left=True)
g.set_xticklabels(rotation=90)
g.set( xticks=range(0,34))
g = g.set_ylabels("Rating")
plt.title('Boîte à moustaches Rating / Category',size = 20)
```




    Text(0.5, 1.0, 'Boîte à moustaches Rating / Category')




![png](output_66_1.png)


#### Le Rating ne different pas beaucoup pour chaque catégory

## Rating / Reviews


```python
plt.figure(figsize = (10,10))
g = sns.jointplot(x="Reviews", y="Rating",color = 'orange', data=data,height = 8);

```


    <Figure size 720x720 with 0 Axes>



![png](output_69_1.png)



```python
plt.figure(figsize = (10,10))
sns.regplot(x="Reviews", y="Rating", color = 'darkorange',data=data[data['Reviews']<1000000]);
plt.title('Rating VS Reveiws',size = 20)
```




    Text(0.5, 1.0, 'Rating VS Reveiws')




![png](output_70_1.png)


#### Il semble que les applications bien connues obtiennent une bonne note

## Rating / Size


```python
plt.figure(figsize = (10,10))
g = sns.jointplot(x="Size", y="Rating",color = 'orangered', data=data, height = 8);
```


    <Figure size 720x720 with 0 Axes>



![png](output_73_1.png)


#### Un commentaire ? 

## Rating / Installs


```python
plt.figure(figsize = (10,10))
sns.regplot(x="Installs", y="Rating", color = 'teal',data=data);
plt.title('Rating VS Installs',size = 20)
```




    Text(0.5, 1.0, 'Rating VS Installs')




![png](output_76_1.png)


#### Il semblerait que le nombre d'installation affecte le Rating

## Rating / Price


```python
plt.figure(figsize = (10,10))
sns.regplot(x="Price", y="Rating", color = 'darkorange',data=data[data['Reviews']<1000000]);
plt.title('Scatter plot Rating VS Price',size = 20)
```




    Text(0.5, 1.0, 'Scatter plot Rating VS Price')




![png](output_79_1.png)


#### Les applications d'un prix plus élevé semble plus décevoir le client. 
#### Pour la suite nous allons créer des fourchettes de prix : 
0  : '0 Gratuit'
0.01 <= 0.99: '1 bas de gamme'
0.99 <= 2.99 : '2 abordables'
2.99 <= 4.99): '3 normale'
4.99 <= 14.99): '4 cher'
14.99 <= 29.99): '5 tres cher'
supérieure à 29.99) : '6 trop cher'


```python
data.loc[ data['Price'] == 0, 'PriceBand'] = '0 Gratuit'
data.loc[(data['Price'] > 0) & (data['Price'] <= 0.99), 'PriceBand'] = '1 bas de gamme'
data.loc[(data['Price'] > 0.99) & (data['Price'] <= 2.99), 'PriceBand']   = '2 abordable'
data.loc[(data['Price'] > 2.99) & (data['Price'] <= 4.99), 'PriceBand']   = '3 normale'
data.loc[(data['Price'] > 4.99) & (data['Price'] <= 14.99), 'PriceBand']   = '4 cher'
data.loc[(data['Price'] > 14.99) & (data['Price'] <= 29.99), 'PriceBand']   = '5 tres cher'
data.loc[(data['Price'] > 29.99), 'PriceBand']  = '6 trop cher'
data[['PriceBand', 'Rating']].groupby(['PriceBand'], as_index=False).mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PriceBand</th>
      <th>Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0 Gratuit</td>
      <td>4.186298</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1 bas de gamme</td>
      <td>4.300943</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2 abordable</td>
      <td>4.292975</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3 normale</td>
      <td>4.250318</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4 cher</td>
      <td>4.269149</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5 tres cher</td>
      <td>4.252000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6 trop cher</td>
      <td>3.923810</td>
    </tr>
  </tbody>
</table>
</div>



#### Les applications bas de gamme entre 0.01 et 0,99 dollars ont les meilleurs notes


```python
g = sns.catplot(x="PriceBand",y="Rating",data=data, kind="boxen", height = 10 ,palette = "Pastel1")
g.despine(left=True)
g.set_xticklabels(rotation=90)
g = g.set_ylabels("Rating")
plt.title('Boxen plot Rating VS PriceBand',size = 20)
```




    Text(0.5, 1.0, 'Boxen plot Rating VS PriceBand')




![png](output_83_1.png)


#### Les prix n'ont pas d'effet sur le Rating , mais pour les applications trop cher le Rating peut etre plus mauvais


## Rating / Content Rating


```python
g = sns.catplot(x="Content Rating",y="Rating",data=data, kind="box", height = 10 ,palette = "Paired")
g.despine(left=True)
g.set_xticklabels(rotation=90)
g = g.set_ylabels("Rating")
plt.title('Box plot Rating VS Content Rating',size = 20)
```




    Text(0.5, 1.0, 'Box plot Rating VS Content Rating')




![png](output_86_1.png)


#### Le classement du contenu n'a pas trop d'effet sur le Rating, mais dans les applications 'Matures', ils ont l'air d'être moins bien notés que les autres.

## Rating / Genres


```python
data[['Genres', 'Rating']].groupby(['Genres'], as_index=False).mean().describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>47.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.210662</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.104405</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3.970769</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4.132039</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.198246</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.282529</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.435556</td>
    </tr>
  </tbody>
</table>
</div>




```python
data[['Genres', 'Rating']].groupby(['Genres'], as_index=False).mean().sort_values('Rating').head(1)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Genres</th>
      <th>Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14</th>
      <td>Dating</td>
      <td>3.970769</td>
    </tr>
  </tbody>
</table>
</div>




```python
data[['Genres', 'Rating']].groupby(['Genres'], as_index=False).mean().sort_values('Rating').tail(1)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Genres</th>
      <th>Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>18</th>
      <td>Events</td>
      <td>4.435556</td>
    </tr>
  </tbody>
</table>
</div>



#### Si l'on observe à partir de l'écart-type, le genre n'a pas trop d'effet sur la notation. La plus faible d'une note moyenne sur les genres (Rencontres) est de 3,97 alors que le plus élevé (Evénements) est de 4,43