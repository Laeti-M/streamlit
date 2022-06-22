import streamlit as st 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
import plotly.express as px 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error
import openpyxl

df=pd.read_csv('df_all.csv')
df=df.drop('Unnamed: 0',axis=1)

options=st.sidebar.radio("Options : " ,['Contexte et objectifs du projet','Présentation des données', 'Visualisations', 'PCA et clustering', 'Méthodes de régression'])
if options == 'Contexte et objectifs du projet' :
    st.image('image.jpg', width=400)
    st.title("Projet Be App'Py 😀") 
    st.markdown("## Contexte du projet")
    st.write("Aujourd’hui le rapport sur le bonheur dans le monde (**World Happiness Report** https://worldhappiness.report/) est une enquête de référence sur l'état du bonheur dans le monde. Le rapport continue de gagner en reconnaissance mondiale, les gouvernements, les organisations et la société civile utilisant de plus en plus les indicateurs de bonheur pour éclairer leurs décisions politiques. Des experts dans divers domaines - économie, psychologie, analyse d'enquêtes, statistiques nationales, santé, politiques publiques et autres - décrivent comment les mesures du bien-être peuvent être utilisées efficacement pour évaluer les progrès des nations. Les rapports passent en revue l'état du bonheur dans le monde d'aujourd'hui et montrent comment la nouvelle science du bonheur explique les variations personnelles et nationales du bonheur.")
    st.markdown("## Objectifs")
    st.write("Deux objectifs ont été définis pour ce projet :"
         "1- utilisation de méthodes de clustering pour déterminer quelles variables jouent un rôle dans le regroupement des pays en fonction du score de bonheur." 
         "2- prédiction du score de bonheur des pays pour l’année 2022 en utilisant des méthodes de régression.")

with st.form('Auteurs :') :
    st.sidebar.markdown('#### Auteurs :')
    link1 = '[LinkedIn](https://www.linkedin.com/in/laetitia-mercey)'
    link2 = '[LinkedIn](https://www.linkedin.com/in/sebastienmorin1/)'
    link3 = '[LinkedIn](https://www.linkedin.com/in/anthony-rizzo-b0403263/)'
    st.sidebar.write('Laetitia Mercey',link1, unsafe_allow_html=True) 
    st.sidebar.write('Sébastien Morin',link2, unsafe_allow_html=True)
    st.sidebar.write('Anthony Rizzo',link3, unsafe_allow_html=True) 
    st.sidebar.markdown("*Formation **DA** format continu OCT21*")
    st.sidebar.markdown("👉 https://datascientest.com/")

if options == 'Présentation des données' :
    st.title("Présentation du jeu de données")
    st.write('Le jeu de données présenté ici, regroupe les données de 2005 à 2021, des scores de bonheur de 160 pays ainsi que plusieurs variables explicatives décrites dans le tableau ci-dessous.')
    st.write('Après nettoyage du dataset, le jeu de données compte 2007 lignes pour 12 colonnes.')
    st.dataframe(df)
    st.markdown("## Explication des variables 🔎")
    var=pd.read_excel('Table_var.xlsx', index_col='N° colonne')
    st.dataframe(var)
    st.caption('Life Ladder (score de bonheur) est la variable cible dans l' ' analyse')
    st.markdown('### *Choisissez un pays* : ')
    pays=st.text_input('', value='')
    st.write(df[df['Country name']==pays])
    
    df2=df[df['Country name']==pays]
    figure = px.line(df2, x="year", y="Life Ladder", title='Life Ladder evolution 📈')
    figure.update_layout(yaxis_range=[2,8])
    st.plotly_chart(figure)


if options == 'Visualisations' :
#ajouter du blabla + autres graph plotly ?
#essayer de modif fig, et fig2 en plotly pour enlever le fond blanc
    st.title("Quelques visualisations")
    st.markdown('### Evolution du score de bonheur en fonction du pays et de l'' année :')
    fig=px.choropleth(df.sort_values('year'),
                  locations="Country name",
                  color="Life Ladder",
                  locationmode="country names",
                  animation_frame="year")
    st.plotly_chart(fig)

    st.markdown('### Boxplot du score de bonheur en fonction des régions du monde :')
    fig1=px.box(df, x='Regional indicator', y='Life Ladder')
    st.plotly_chart(fig1)

    st.markdown('### Top 5 des pays les plus heureux en moyenne depuis 2005 (a gauche), les moins heureux en moyenne depuis 2005 (a droite) : ')
    fig2=plt.figure(figsize=(10, 5))
    df2 = pd.DataFrame(df.groupby("Country name").mean())
    df2 = df2.reset_index()
    df2 = df2.sort_values('Life Ladder', ascending=False)
    plt.subplot(121)
    sns.barplot(y=df2["Life Ladder"].head(10), x=df2["Country name"].head(10))
    plt.xticks(rotation=50, ha='right')
    plt.subplot(122)
    sns.barplot(y=df2["Life Ladder"].tail(10), x=df2["Country name"].tail(10))
    plt.axis([-0.5,9.5,0,8.2])
    plt.xticks(rotation=50, ha='right');
    st.pyplot(fig2)
    
    
if options == 'PCA et clustering':  
    st.title('PCA et clustering')
    st.write("""
    L'analyse en composantes principales (PCA pour Principal Component Analysis) est une méthode de réduction 
    de dimension qui consiste à réduire la complexité superflue d'un jeu de données en projetant 
    ses données dans un espace de plus petite dimension.
             """)
    with st.expander("En savoir plus sur la PCA :"):
        st.write("""
    Il s’agit de résumer l’information contenue dans un ensemble de données en un certain nombre de variables synthétiques, combinaisons linéaires des variables originelles : ce sont les Composantes Principales.
    L’enjeu est généralement de réduire de manière significative la dimension du jeu de données tout
    en conservant au maximum l'information véhiculée par les données. On parle de part de variance expliquée.
    Le but ici est de permettre d'accélérer l'apprentissage et de réduire le risque d'overfitting lié au 
    surplus de dimensions.
    """)
        st.markdown("👉 https://fr.wikipedia.org/wiki/Analyse_en_composantes_principales")
    st.write("""
    Une tâche fréquente en analyse de données consiste, à partir d'un ensemble d'observations,
    à créer des groupes d'individus (clusters) de telle sorte que les individus d'un groupe donné aient tendance à être similaires, 
    et en même temps aient tendance à être différents des individus des autres groupes.
    Les algorithmes de classification non supervisée répondent à cette tâche, ils utilisent un ensemble de données non-étiquetées ou non-labellisées et recherchent les structures naturelles dans les données.
             """)
    with st.expander("En savoir plus sur le clustering :"):
        st.write("""
    L'algorithme K-Means vise à diviser tous les points du jeu de données en k groupes, appelés clusters, homogènes et compacts. 
    Pour ce faire il va répéter les opérations suivantes pour obtenir les meilleurs résultats :
    - Choisir k centroïdes aléatoirement
    - Calculer les distances avec les k-centroïdes pour chaque point du dataset
    - Assigner chaque point au centroïde le plus proche
    - Actualiser les centroïdes comme centre des nouveaux cluster obtenus
    """)
        st.markdown("👉 https://fr.wikipedia.org/wiki/K-moyennes")
        
    st.markdown('### Méthodes : ')
    st.write('La PCA a été appliqué sur tout le dataset, excepté la variable cible (Life Ladder) et la variable country name.')
    st.write('KMeans a ensuite été utilisé sur les données issues de la PCA, pour visualiser les 4 clusters créés en 2D, visibles ci-dessous.')
    st.info('Après application de la PCA et en utilisant uniquement les 2 premières composantes principales, le % de variance expliquée conservée est de 36.5.')
    
    
# préparation des données
    target=df['Life Ladder']
    data=df.drop(['Life Ladder','Country name'], axis=1)
    data_dummies=pd.get_dummies(data)
    scaler=StandardScaler()
    data_scaled=scaler.fit_transform(data_dummies)
    pca=PCA()
    pca=pca.fit(data_scaled)
    coord=pca.transform(data_scaled)
    df_coord=pd.DataFrame(coord)
    kmeans=KMeans(n_clusters=4, random_state=50)
    kmeans.fit(df_coord)
    y_kmeans=kmeans.predict(df_coord)
    
    plt.figure(figsize=(15,10))
    plt.scatter(df_coord[y_kmeans==0].iloc[:,0], df_coord[y_kmeans==0].iloc[:,1],
            s=50, c='plum', label='Cluster 1')
    plt.scatter(df_coord[y_kmeans==1].iloc[:,0], df_coord[y_kmeans==1].iloc[:,1],
            s=50, c='skyblue', label='Cluster 2')
    plt.scatter(df_coord[y_kmeans==2].iloc[:,0], df_coord[y_kmeans==2].iloc[:,1], 
            s=50, c='chocolate', label='Cluster 3')
    plt.scatter(df_coord[y_kmeans==3].iloc[:,0], df_coord[y_kmeans==3].iloc[:,1], 
            s=50, c='gold', label='Cluster 4')
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='black', marker='s', s=30)
    plt.legend()
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    df_cluster = df.copy()
    df_cluster['cluster_kmeans']=y_kmeans
    cluster_1=df_cluster[df_cluster['cluster_kmeans']==0]
    cluster_2=df_cluster[df_cluster['cluster_kmeans']==1]
    cluster_3=df_cluster[df_cluster['cluster_kmeans']==2]
    cluster_4=df_cluster[df_cluster['cluster_kmeans']==3]
       
    st.markdown('### *Choix du n° de cluster* : ')
    number_cluster=st.selectbox('', [1,2,3,4])
        
    if number_cluster == 1 :
        fig=plt.figure(figsize = (10,5))
        spec = gridspec.GridSpec(ncols=2, nrows=1,
                         width_ratios=[3,5.5])
        fig.subplots_adjust(wspace=1)
        ax1=fig.add_subplot(spec[0])
        x=cluster_1['Regional indicator'].value_counts()
        ax1.pie(x, labels=['Sub-Saharan Africa', 'South Asia', 'Latin America, Caribbean'], autopct='%1.2f%%', 
                pctdistance = 0.8, radius=3, labeldistance=None,
                colors=['orange', 'green', 'purple'])
        ax1.legend(loc='upper center')
        
        ax1=fig.add_subplot(spec[1])
        ax1.boxplot(cluster_1[['Life Ladder', 'Log GDP per capita']], 
        labels=['Life Ladder', 'Log GDP per capita'], boxprops= dict(linewidth=1, color='blue'), notch=True)
        ax1.set_ylabel('Life Ladder, Log GDP per capita', color='blue')
        ax2 = ax1.twinx()
        ax2.boxplot(cluster_1[['Social support','Freedom to make life choices', 'Generosity',
       'Perceptions of corruption', 'Positive affect', 'Negative affect']], labels=['Social support',
       'Freedom to make life choices', 'Generosity','Perceptions of corruption', 'Positive affect',
        'Negative affect'], boxprops= dict(linewidth=1, color='seagreen'), positions=[3, 4, 5, 6, 7, 8],notch=True)
        ax2.set_ylabel('Social, Freedom, Generosity, corruption, pos/neg affect', color='seagreen')
        plt.setp(ax1.get_xticklabels(), rotation=30, ha="right")
        st.pyplot(fig)
        
        st.write("""
        Le cluster n°1 (a droite sur le graphique), correspond majoritairement aux pays de la région Afrique sub-saharienne.
        On peut observer avec les boxplots, que ce cluster regroupe les pays avec les scores les plus faibles de bonheur,
        ainsi que pour le PIB/hab. On retrouve une grande hétérogénéité pour la variable
        corruption avec beaucoup d'outliers, tout comme pour générosité ou même le PIB.
                 """)
        
    elif number_cluster == 2 :
        fig=plt.figure(figsize = (10,5))
        spec = gridspec.GridSpec(ncols=2, nrows=1,
                         width_ratios=[3,5.5])
        fig.subplots_adjust(wspace=1)
        ax1=fig.add_subplot(spec[0])
        x=cluster_2['Regional indicator'].value_counts()
        ax1.pie(x, labels=['Western Europe', 'North America and ANZ ', 'Middle East and North Africa', 'Southeast Asia', 'Central and Eastern Europe'],
        colors=['royalblue', 'grey', 'tan', 'red', 'darkturquoise'], 
        autopct='%1.2f%%', pctdistance = 0.8, radius=3, labeldistance=None)
        ax1.legend(loc='upper center')
        
        ax1=fig.add_subplot(spec[1])
        ax1.boxplot(cluster_2[['Life Ladder', 'Log GDP per capita']], 
            labels=['Life Ladder', 'Log GDP per capita'], boxprops= dict(linewidth=1, color='blue'), notch=True)
        ax1.set_ylabel('Life Ladder, Log GDP per capita', color='blue')
        ax2 = ax1.twinx()
        ax2.boxplot(cluster_2[['Social support','Freedom to make life choices', 'Generosity',
       'Perceptions of corruption', 'Positive affect', 'Negative affect']], labels=['Social support',
       'Freedom to make life choices', 'Generosity','Perceptions of corruption', 'Positive affect',
        'Negative affect'], boxprops= dict(linewidth=1, color='seagreen'), positions=[3, 4, 5, 6, 7, 8],notch=True)
        ax2.set_ylabel('Social, Freedom, Generosity, corruption, pos/neg affect', color='seagreen')
        plt.setp(ax1.get_xticklabels(), rotation=30, ha="right")
        st.pyplot(fig);
        
        st.write("""
        Le cluster n°2 (a gauche sur le graphique), correspond majoritairement aux pays de la région Europe de l'Ouest.
        On peut observer avec les boxplots, que ce cluster regroupe les pays avec les scores les plus hauts de bonheur,
        ainsi que pour le PIB/hab. Les variables support social et liberté de faire des choix sont elles aussi très élévées
        par rapport aux autres clusters, avec une distribution beaucoup moins étendue mis a part plusieurs outliers.
                 """)
        
    elif number_cluster == 3 :
        fig=plt.figure(figsize = (10,5))
        spec = gridspec.GridSpec(ncols=2, nrows=1,
                         width_ratios=[3,5.5])
        fig.subplots_adjust(wspace=1)
        ax1=fig.add_subplot(spec[0])
        x=cluster_3['Regional indicator'].value_counts()
        ax1.pie(x,labels=['Central and Eastern Europe', 'Middle East and North Africa', 'South Asia', 'Western Europe'],
        colors=['darkturquoise', 'tan', 'green', 'royalblue'],
        autopct='%1.2f%%', pctdistance = 0.8, radius=3, labeldistance=None)
        ax1.legend(loc='upper center')
        
        ax1=fig.add_subplot(spec[1])
        ax1.boxplot(cluster_3[['Life Ladder', 'Log GDP per capita']], 
        labels=['Life Ladder', 'Log GDP per capita'], boxprops= dict(linewidth=1, color='blue'), notch=True)
        ax1.set_ylabel('Life Ladder, Log GDP per capita', color='blue')
        ax2 = ax1.twinx()
        ax2.boxplot(cluster_3[['Social support','Freedom to make life choices', 'Generosity',
       'Perceptions of corruption', 'Positive affect', 'Negative affect']], labels=['Social support',
       'Freedom to make life choices', 'Generosity','Perceptions of corruption', 'Positive affect',
        'Negative affect'], boxprops= dict(linewidth=1, color='seagreen'), positions=[3, 4, 5, 6, 7, 8],notch=True)
        ax2.set_ylabel('Social, Freedom, Generosity, corruption, pos/neg affect', color='seagreen')
        plt.setp(ax1.get_xticklabels(), rotation=30, ha="right")
        st.pyplot(fig);
        
        st.write("""
        Le cluster n°3 (au milieu et en bas sur le graphique), correspond majoritairement aux pays de la région Europe centrale et Europe de l'Est.
        Ce cluster correspond a des pays qui ont un score de bonheur moyen et pour lequel 
        la distribution des différentes variables est proche du cluster 4 (au dessus de lui).
                 """)

    elif number_cluster == 4 :
        fig=plt.figure(figsize = (10,5))
        spec = gridspec.GridSpec(ncols=2, nrows=1,
                         width_ratios=[3,5.5])
        fig.subplots_adjust(wspace=1)
        ax1=fig.add_subplot(spec[0])
        x=cluster_4['Regional indicator'].value_counts()
        ax1.pie(x,labels=['Latin America and Caribbean', 'Southeast Asia', 'South Asia', 'East Asia', 'Sub-Saharan Africa'],
        colors=['purple', 'red', 'green', 'pink', 'orange'],
        autopct='%1.2f%%', pctdistance = 0.8, radius=3, labeldistance=None)
        ax1.legend(loc='upper center')

        ax1=fig.add_subplot(spec[1])
        ax1.boxplot(cluster_4[['Life Ladder', 'Log GDP per capita']], 
            labels=['Life Ladder', 'Log GDP per capita'], boxprops= dict(linewidth=1, color='blue'), notch=True)
        ax1.set_ylabel('Life Ladder, Log GDP per capita', color='blue')
        ax2 = ax1.twinx()
        ax2.boxplot(cluster_4[['Social support','Freedom to make life choices', 'Generosity',
       'Perceptions of corruption', 'Positive affect', 'Negative affect']], labels=['Social support',
       'Freedom to make life choices', 'Generosity','Perceptions of corruption', 'Positive affect',
        'Negative affect'], boxprops= dict(linewidth=1, color='seagreen'), positions=[3, 4, 5, 6, 7, 8],notch=True)
        ax2.set_ylabel('Social, Freedom, Generosity, corruption, pos/neg affect', color='seagreen')
        plt.setp(ax1.get_xticklabels(), rotation=30, ha="right")
        st.pyplot(fig);
        
        st.write("""
        Le cluster n°4 (au milieu et en haut sur le graphique), correspond majoritairement aux pays de la région Amérique latine et Caraïbes.
        Ce cluster correspond a des pays qui ont un score de bonheur moyen et pour lequel 
        la distribution des différentes variables est proche du cluster 3 (en dessous de lui).
        Ce cluster présente la plus grande hétérogénéité pour la variable générosité et beaucoup d'outliers, indiquant qu'au sein
        de ce cluster il y a de grandes différences pour ce paramètre. Il faut noter que ce cluster est réparti pour
        5 régions mondiales et de façon moins nette que pour le cluster 1 et 2 ce qui explique cette hétérogénéité.  
                 """)

if options == 'Méthodes de régression' : 
    st.title('Méthodes de régression')
    st.write('Dans le cadre de notre partie concernant la prédiction des niveaux de bonheur des pays concernés par l’étude pour l’année 2022, notre variable cible étant d’ordre numérique, nous avons orienté notre sélection d’algorithmes vers les **méthodes de régression**. Nous avons choisi de tester l’ensemble des modèles de ce genre qui nous ont été présentés lors de notre parcours pour pouvoir ensuite les comparer, en l’occurrence les modèles de **régression linéaire, RidgeCV, LassoCV ainsi qu’ElasticNetCV**.')
    st.write('')
    
    df_predict = df[["Country name","year","Life Ladder"]]
    df_predict["year"] = df_predict["year"]-1
    df_new = df.drop(["Life Ladder"], axis = 1)
    dfsuperpredict = df_new.merge(df_predict, on = ["Country name", "year"], how = "inner")
    dfsuperpredict2=pd.get_dummies(dfsuperpredict)
    dfsuperpredict2.sort_values(by = "year", ascending = True)

    target=dfsuperpredict2['Life Ladder']
    data=dfsuperpredict2.drop(['Life Ladder'], axis=1)
    X_train, X_test, y_train, y_test=train_test_split(data, target, test_size=0.2, random_state=100)
    scaler=StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)

    ridge_reg = RidgeCV(alphas= (0.001, 0.01, 0.1, 0.3, 0.7, 1, 10, 50, 100))
    ridge_reg.fit(X_train_scaled, y_train) 

    df_2021=pd.read_csv('df_2021.csv')
    df_2021=df_2021.drop('Unnamed: 0', axis=1)
    df_2021_dummies=pd.get_dummies(df_2021)
    df_2021_scaled=scaler.transform(df_2021_dummies)
    df_2022_predict= ridge_reg.predict(df_2021_scaled)
    dataname = list(df_2021["Country name"])
    df_2022=pd.DataFrame({'Bonheur 2022': df_2022_predict}, index=dataname)

# Choix pays --> score de bonheur
    st.markdown('### *Choisissez un pays pour afficher la prévision de son score de bonheur en 2022* : ')
    bonheur=st.text_input('', value='')
    st.write(df_2022[df_2022.index==bonheur])
    st.write('')
    
# carte du monde et prédiction 2022   
    fig=px.choropleth(df_2022,
                  locations=df_2022.index,
                  color="Bonheur 2022",
                  locationmode="country names", title='Carte des prédictions du score de bonheur 2022')
    fig.update_layout(title_x=0.5)
    st.plotly_chart(fig)
