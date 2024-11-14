import streamlit as st
from matplotlib import _preprocess_data
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, max_error, mean_absolute_error, r2_score
import warnings
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from PIL import Image
warnings.filterwarnings('ignore')

# Configuration de la page principale
st.set_page_config(
    page_title="Projet de Prédiction du Score de Bonheur",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="collapsed"
)
st.markdown("<h1 style='text-align: center;'>✨Projet de Prédiction du Score de Bonheur✨</h1>", unsafe_allow_html=True)

st.sidebar.title("🧭 Navigation")
pages = {
    "Page de bienvenue": "Explorez le projet et comprenez ses objectifs.",
    "Pré-traitement des données": "Détails sur la préparation et le nettoyage des données.",
    "Visualisation": "Découvrez les graphiques et les insights visuels.",
    "Modélisation": "Examinez les modèles utilisés et leurs performances.",
    "Machine Learning": "Explorez l'implémentation des algorithmes.",
    "Conclusions et Perspectives": "Résumé et discussion sur les perspectives.",
    "Auteurs": "Présentation de l'équipe du projet."
}

# Initialisation de la page sélectionnée dans la session Streamlit
if 'selected_page' not in st.session_state:
    st.session_state.selected_page = None

# Affichage des boutons de navigation et descriptions dans la barre latérale
for page, description in pages.items():
    if st.sidebar.button(page):
        st.session_state.selected_page = page
    # Affichage de la description sous chaque bouton
    st.sidebar.write(f"ℹ️ {description}")

# Ajout du bouton "Retour"
if st.sidebar.button("⬅️ Retour"):
    st.session_state.selected_page = None

# Chargement des données et préparation des fonctions de pages
@st.cache_data
def charger_donnees():
    df = pd.read_csv('world-happiness-report.csv')
    df2 = pd.read_csv('world-happiness-report-2021.csv')
    df_energy = pd.read_csv('owid-energy-data.csv')
    return df, df2, df_energy

df, df2, df_energy = charger_donnees()

@st.cache_data
def data_preparation(df, df2, df_energy):

    df=df.merge(df2[['Country name','Regional indicator']],on='Country name',how='left')

    dictionnaire = {'Ladder score': 'Life Ladder',
                        'Logged GDP per capita': 'Log GDP per capita',
                        'Social support'      : 'Social support',
		            	      'Healthy life expectancy'      : 'Healthy life expectancy at birth',
			                  'Freedom to make life choices'      : 'Freedom to make life choices',
			                  'Generosity'      : 'Generosity',
                        'Perceptions of corruption'       : 'Perceptions of corruption'}

    df2= df2.rename(dictionnaire, axis = 1)

    df3 = df2.drop(['Standard error of ladder score', 'upperwhisker', 'lowerwhisker','Ladder score in Dystopia','Explained by: Log GDP per capita','Explained by: Social support','Explained by: Healthy life expectancy',
                'Explained by: Freedom to make life choices','Explained by: Generosity','Explained by: Perceptions of corruption','Dystopia + residual'], axis=1)

    df3['year']=2021
    df3.head()

    df4 = pd.concat([df, df3], ignore_index=True)

    dictionnaire = {'Life Ladder': 'Life_Ladder',
                        'Log GDP per capita': 'Log_GDP_per_capita',
                        'Social support'      : 'Social_support',
		            	      'Healthy life expectancy at birth'      : 'Healthy_life_expectancy_at_birth',
			                  'Freedom to make life choices'      : 'Freedom_to_make_life_choices',
			                  'Generosity'      : 'Generosity',
                        'Perceptions of corruption  '       : 'Perceptions_of_corruption'}

    df4 = df4.rename(dictionnaire, axis = 1)
   
    df_energy.rename(columns ={'country' : 'Country name'}, inplace = True)

    df_energy = df_energy[['year','Country name','population', 'energy_per_capita']]

    df_final = df4.merge(df_energy, on=['Country name', 'year'], how='inner')
    df_final = df_final[df_final['year'] != 2005]

    region_dict = {
    'Angola': 'Sub-Saharan Africa',
    'Central African Republic': 'Sub-Saharan Africa',
    'Djibouti': 'Sub-Saharan Africa',
    'Somalia': 'Sub-Saharan Africa',
    'South Sudan': 'Sub-Saharan Africa',
    'Sudan': 'Sub-Saharan Africa',
    'Belize': 'Latin America and Caribbean',
    'Cuba': 'Latin America and Caribbean',
    'Guyana': 'Latin America and Caribbean',
    'Suriname': 'Latin America and Caribbean',
    'Trinidad and Tobago': 'Latin America and Caribbean',
    'Bhutan': 'South Asia',
    'Oman': 'Middle East and North Africa',
    'Qatar': 'Middle East and North Africa',
    'Syria': 'Middle East and North Africa'}

    df_final['Regional indicator'] = df_final['Country name'].map(region_dict).fillna(df_final['Regional indicator'])

    df_final['Positive affect'] = df_final['Positive affect'].fillna(0)
    df_final['Negative affect'] = df_final['Negative affect'].fillna(0)

    return df_final

df_final = data_preparation(df, df2, df_energy)

@st.cache_data
def page_pretraitement_donnees(df_final):

    feature = df_final.drop(columns=['Life_Ladder'])
    target = df_final['Life_Ladder']

    X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, random_state=42)

    numeric_features = ['Log_GDP_per_capita', 'Social_support', 'Healthy_life_expectancy_at_birth',
                'Freedom_to_make_life_choices', 'Generosity', 'Perceptions of corruption',
                'Positive affect', 'Negative affect', 'population', 'energy_per_capita']

    imputer = SimpleImputer(missing_values = np.nan, strategy = 'median') 

    X_train[numeric_features] = imputer.fit_transform(X_train[numeric_features])
    X_test[numeric_features] = imputer.transform(X_test[numeric_features])

    cat_features = ['Regional indicator']
    cat_train = X_train[cat_features]
    cat_test = X_test[cat_features]

    oneh = OneHotEncoder (drop = 'first', sparse_output = False, handle_unknown = 'ignore')

    encoded_train = oneh.fit_transform(X_train[cat_features])
    encoded_test = oneh.transform(X_test[cat_features])

    encoded_cat_train = pd.DataFrame(encoded_train, columns = oneh.get_feature_names_out(cat_train.columns))
    encoded_cat_test = pd.DataFrame(encoded_test, columns = oneh.get_feature_names_out(cat_test.columns))

    numeric_train = X_train[numeric_features]
    numeric_test = X_test[numeric_features]

    scaler = StandardScaler()

    numeric_train_scaled = scaler.fit_transform(numeric_train)
    numeric_test_scaled = scaler.transform(numeric_test)

    numeric_train_scaled = pd.DataFrame(numeric_train_scaled, columns=numeric_features)
    numeric_test_scaled = pd.DataFrame(numeric_test_scaled, columns=numeric_features)
    
    X_train_final = pd.concat([numeric_train_scaled, encoded_cat_train], axis=1)
    X_test_final = pd.concat([numeric_test_scaled, encoded_cat_test], axis=1)

    return X_train_final, X_test_final, y_train, y_test

X_train_final, X_test_final, y_train, y_test = page_pretraitement_donnees(df_final)

# Affichage de la page sélectionnée
if st.session_state.selected_page:
    st.header(f" {st.session_state.selected_page}")

    # Contenu de chaque page
    if st.session_state.selected_page == "Page de bienvenue":
        st.write("### Bienvenue sur le projet de prédiction du score de bonheur")
        st.write("""
        Au cours de ce projet nous allons effectuer une analyse approfondie des données collectées 
        par le **World Happiness Report** afin de répondre à une question qui pourrait paraître existentielle :
        “qu’est-ce que le bonheur ?” ou alors “qu’est ce qui me rend heureux ? 
        Pas facile de répondre à cette question, vous me direz, nous allons donc tourner cette question d’une 
        autre manière qui est la suivante : “Le bonheur est-il mesurable ?”

        ### Contexte du projet
        Ce projet a pour objectif de prédire le **score de bonheur** des pays en s'appuyant sur des données socio-économiques et énergétiques obtenus depuis Kaggle.
        Notre analyse s'est concentrée sur l'influence de différents indicateurs, tels que :
        - Le Produit Intérieur Brut (PIB),
        - L'espérance de vie,
        - Le support social,
        - La consommation d'énergie par habitant,
        - Et d'autres indicateurs régionaux.
         
        ### Objectifs du projet
        - **Analyser** les corrélations entre différents indicateurs et le bien-être des populations.
        - **Développer des modèles prédictifs** performants en testant plusieurs algorithmes de machine learning.
        - **Optimiser les modèles choisis** pour garantir des prédictions robustes et adaptées à des études socio-économiques futures.

        ### Méthodologie
        Nous avons testé cinq modèles initiaux :
        - Régression Linéaire
        - Support Vector Regressor (SVR)
        - Random Forest Regressor
        - Gradient Boosting Regressor (GBR)
        - XGBoost Regressor
        """)
        st.write("Nous espérons que ce projet vous inspirera pour explorer de nouvelles pistes d'analyse socio-économiques. Bonne découverte!")

    elif st.session_state.selected_page == "Pré-traitement des données":
        st.write("""
        #### Ci dessous les étapes de preprocessing  appliquées à notre jeu de données
        
        ###### Le dataset original de Kaggle
        Nous avons décidé de retirer l'année 2005, car les données sont limitées cette année là ( nombre de pays sondés très faible comparé aux autres années)""") 
        
        if st.checkbox("Afficher échantillon du dataset original pour l'année 2021"):
            st.write(df2.head(10))


        st.write("""
        ###### Ajout de données externes
        Enrichissement avec deux variables explicatives externes recherchées sur des dataset externes, qui nous ont semblées intéressantes quant à leur impact sur le score du bonheur : population du pays et  niveau d’énergie consommé par habitant par pays

                 
        **Notre jeu de données final**, après combinaison des données précitées, se compose de 
        - la variable cible : Life Ladder, représentant le score de bonheur à prédire.
        - 13 variables explicatives, sélectionnées pour leur pertinence""") 
        # Créer un buffer pour capturer la sortie de df.info()
        import io
        buffer = io.StringIO()
        df_final.info(buf=buffer)
        info = buffer.getvalue()
        st.text(info)

        st.write("""
        ###### Transformation
        Nous avons identifié les transformations suivantes, qui ont été appliquées après le train test split afin d'éviter la fuite de données
        - Vérification des doublons
        - Remplacement des valeurs manquantes par la médiane , pour les variables numériques
        - Encodage de la variable  catégorielle "Regional Indicator"
        - Normalisation des variables continues""") 

        st.write("Les jeux d'entrainement et de test sont prêts pour l'étape de modélisation")

        st.write(X_train_final.head(10))

        
    elif st.session_state.selected_page == "Visualisation":
        st.write("### Découvrez les visualisations de données")
        st.write("Explorez les différentes représentations visuelles des données pour mieux comprendre leurs relations et distributions.")
        
        # Menu déroulant pour choisir la visualisation
        option = st.selectbox(
        "Sélectionnez une visualisation",
        ("Carte Choroplèthe", "Matrice de corrélation", "Distribution de la variable cible", "Boxplot de la distribution des variables", "Top & Worst 5"))
    
        # Carte Choroplèthe
        if option == "Carte Choroplèthe":
            st.subheader("Carte Choroplèthe")
            st.write("Pour illustrer l'évolution du score de bonheur à travers les pays et sur plusieurs années, nous avons utilisé une carte choroplèthe. Cette visualisation permet de représenter les disparités géographiques de bien-être à l’échelle mondiale : ")
            fig = px.choropleth(df_final.sort_values("year"), locations="Country name", locationmode="country names", color="Life_Ladder",
                             animation_frame = "year", title="Score de bonheur par pays")
            st.plotly_chart(fig)
            st.write("""
                     On y observe plusieurs tendances intéressantes :

                     - **États-Unis** : diminution progressive du score de bonheur au fil des années.
                     - **Australie** :  score relativement stable et globalement positif.
                     - **Russie**, Asie centrale et Asie :  scores moyens qui restent également stables au cours du temps.
                     - **Afrique** :  amélioration lente mais continue du score de bonheur au fil des années.
                     - **Europe** : Scores globalement élevés et stables, avec des résultats particulièrement remarquables pour les pays scandinaves.
                     
                     L'année 2013 se distingue de façon positive pour l'ensemble des pays.
                     Enfin, on note une légère baisse des scores de bonheur en 2020, probablement en raison de la pandémie de Covid-19.
                     """)

        # Matrice de corrélation
        elif option == "Matrice de corrélation":
            st.subheader("Matrice de corrélation des variables explicatives")
            st.write("""Un problématique crucial est de vérifier si certaines variables explicatives présentent une forte corrélation entre elles, ce qui pourrait entraîner des problèmes de **multicolinéarité** lors de la modélisation. 
Une multicolinéarité élevée pourrait non seulement nuire à la précision du modèle, mais aussi compliquer l'interprétation des résultats. Pour explorer cette hypothèse, nous allons visualiser **une matrice de corrélation (heatmap)** :
""")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(X_train_final.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            st.pyplot(fig)
            st.write("""Nous avons constaté **une forte corrélation entre plusieurs variables explicatives**, notamment le PIB par habitant, l'espérance de vie et le soutien social. 
                     Cette interdépendance peut entraîner des problèmes de **multicolinéarité** lors de la modélisation.""")

        # Distribution de la variable cible
        elif option == "Distribution de la variable cible":
            st.subheader("Distribution de la variable cible - Score de bonheur (Life Ladder)")
            st.write("""La distribution de la variable cible **Life Ladder** nous permets de déterminer 
                     si les extrêmes diminuent et si les scores se resserrent davantage autour de la médiane, 
                     ce qui pourrait indiquer une convergence progressive des niveaux de bonheur entre les pays. 
                     Globalement, **le score de bonheur se situe entre 4,6 et 7,2**. 
                     """)
            fig, ax = plt.subplots()
            sns.histplot(df_final['Life_Ladder'], kde=True, bins=20, color='skyblue', ax=ax)
            st.pyplot(fig)

        # Boxplot de la distribution des variables explicatives
        elif option == "Boxplot de la distribution des variables":
            st.subheader("Boxplot de la distribution des variables explicatives")
            st.write("""Nous observons que certains variables explicatives suivent des tendances similaires,
                      ce qui pourrait indiquer **une relation étroite entre elles**. 
                     Cela suggère que les variables comme le PIB pourrait jouer un rôle prépondérant dans la prédiction du bonheur. """)
            numeric_features = ['Log_GDP_per_capita', 'Social_support', 'Healthy_life_expectancy_at_birth',
                'Freedom_to_make_life_choices', 'Generosity', 'Perceptions of corruption',
                'Positive affect', 'Negative affect', 'population', 'energy_per_capita']
            features = X_train_final[numeric_features]
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(data=features, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            st.pyplot(fig)
                     
        # Composition des régions
        elif option == "Top & Worst 5":
            st.subheader("Pays avec le meilleur et le pire score de bonheur")
            st.write("Nous avons également identifié les pays ayant les **meilleurs** et les **pires scores de bonheur** en moyenne au fil des années : ")
            df_grouped = df_final.groupby('Country name')['Life_Ladder'].mean().reset_index()
            df_sorted = df_grouped.sort_values('Life_Ladder', ascending = False)
            top_5 = df_sorted.head(5)
            worst_5 = df_sorted.tail(5)
            top_worst = pd.concat([top_5, worst_5])
            colors = ['green'] * len(top_5) + ['red'] * len(worst_5)
            
            fig = go.Figure(go.Bar(x=top_worst['Life_Ladder'], y=top_worst['Country name'], orientation='h', marker_color=colors))
            fig.update_layout(title="Pays avec le meilleur et le pire score de bonheur", xaxis_title="Score de bonheur", yaxis_title="Pays", showlegend=False)

            st.plotly_chart(fig)
            st.write("Ce graphique nous montre **un écart très important entre les pays**, qui doit donc pouvoir s’expliquer par l’ensemble des variables mis à notre disposition dans le data frame")

    elif st.session_state.selected_page == "Modélisation":
        st.subheader("Présentation des modèles et leur performance")
        st.write("""Nous avons choisi cinq modèles pour prédire le score du bonheur, dont la nature du problème est de type régression. 
                \n Les modèles sont ensuite évalués à l'aide des métriques standards :
                \n -**MAE** (Mean Absolute Error) : Mesure l’erreur moyenne entre les valeurs réelles et les prédictions sans tenir compte du signe.
                \n -**MSE** (Mean Squared Error) : Donne une importance plus grande aux erreurs importantes en élevant chaque différence au carré.
                \n -**RMSE** (Root Mean Squared Error) : Indique l'écart type des résidus, offrant une interprétation plus facile de la précision du modèle en utilisant les mêmes unités que la variable cible.
                \n -**R²** : Représente la proportion de variance de la variable cible expliquée par les variables explicatives.
                """)
        st.text("")
        st.text("")
        st.text("")
        # Menu déroulant pour choisir la visualisation
        option = st.selectbox(
        "**Sélectionnez un modele**",
        ("Régression linéaire", "Support Vector Regression SVR", "Random Forest Regressor", "Gradient Boosting Regressor", "XGboost"))
        st.text("")
        st.text("")

        # Régression linéaire
        if option == "Régression linéaire":
            st.write("**Régression linéaire**")
            st.write("Un modèle simple et interprétable qui suppose une relation linéaire entre les variables explicatives et la variable cible du bien être à l’échelle mondiale : ")
            st.write(""" **Résultats**:
                    \n - Score train : 0.7995
                    \n - Score test : 0.7812
                    \n - MSE : 0.2648
                    \n - MAE : 0.3952
                    \n - RMSE : 0.5146
                     """)
            st.text("")
            st.write("""**Graphique sur les valeurs réelles versus valeurs prédites (gauche)**
                     \n **Graphique sur les Feature Importance de régression linéaire (droite)**""")
            # Afficher l'image dans Streamlit
            FI_regression_lineaire = Image.open("FI_regression_lineaire.png")
            st.image(FI_regression_lineaire, use_column_width=True)
            st.text("")
            st.text("")
            st.text("")
            st.text("")

        # Support Vector Regression SVR
        elif option == "Support Vector Regression SVR":
            st.write("**Support Vector Regression SVR**")
            st.write("Un modèle plus flexible, basé sur les marges, qui est particulièrement efficace dans les situations où la relation entre les variables n'est pas strictement linéaire.")
            st.write(""" Résultats:
                    \n - Score train : 0.8924
                    \n - Score test : 0.8634
                    \n - MSE : 0.1653
                    \n - MAE : 0.3107
                    \n - RMSE : 0.4066
                     """)
            st.text("")
            st.write("""**Graphique sur les valeurs réelles versus valeurs prédites**
                     \n ****""")
            # Afficher l'image dans Streamlit
            FI_SVR = Image.open("FI_SVR.png")
            st.image(FI_SVR, use_column_width=True)
            st.text("")
            st.text("")
            st.text("")
            st.text("")

        # Random Forest Regressor
        elif option == "Random Forest Regressor":
            st.write("**Random Forest Regressor**")
            st.write("Un modèle basé sur des ensembles d'arbres de décision, robuste aux données bruitées et efficace pour capturer les interactions non linéaires entre les variables")
            st.write(""" Résultats:
                    \n - Score train : 0.9858
                    \n - Score test : 0.8923
                    \n - MSE : 0.1304
                    \n - MAE : 0.2718
                    \n - RMSE : 0.3611
                     """)
            st.text("")
            st.write("""**Graphique sur les valeurs réelles versus valeurs prédites (gauche)**
                     \n **Graphique sur les Feature Importance de Random Forest (droite)**""")
            # Afficher l'image dans Streamlit
            FI_RForest = Image.open("FI_RForest.png")
            st.image(FI_RForest, use_column_width=True)
            st.text("")
            st.text("")
            st.text("")
            st.text("")

        # Gradient Boosting Regressor
        elif option == "Gradient Boosting Regressor":
            st.write("**Gradient Boosting Regressor**")
            st.write("Un modèle basé sur des arbres de décision successifs, où chaque nouvel arbre corrige les erreurs des précédents, ce qui le rend efficace pour réduire les erreurs résiduelles de manière itérative")
            st.write(""" Résultats:
                    \n - Score train : 0.9085
                    \n - Score test : 0.8580
                    \n - MSE : 0.1718
                    \n - MAE : 0.3212
                    \n - RMSE : 0.4145
                     """)
            st.text("")
            st.write("""**Graphique sur les valeurs réelles versus valeurs prédites (gauche)**
                     \n **Graphique sur les Feature Importance de Gradient Boosting (droite)**""")
            # Afficher l'image dans Streamlit
            FI_GBR = Image.open("FI_GBR.png")
            st.image(FI_GBR, use_column_width=True)
            st.text("")
            st.text("")
            st.text("")
            st.text("")
                     
        # XGboost
        elif option == "XGboost":
            st.write("**XGboost**")
            st.write("Une version optimisée du Gradient Boosting, plus rapide et souvent plus performante grâce à son utilisation d'optimisations comme la régularisation")
            st.write(""" Résultats:
                    \n - Score train : 0.9990
                    \n - Score test : 0.8851
                    \n - MSE : 0.1391
                    \n - MAE : 0.2798
                    \n - RMSE : 0.3729
                     """)
            st.text("")
            st.write("""**Graphique sur les valeurs réelles versus valeurs prédites (gauche)**
                     \n **Graphique sur les Feature Importance de XGBoost (droite)**""")
            # Afficher l'image dans Streamlit
            FI_XGB = Image.open("FI_XGB.png")
            st.image(FI_XGB, use_column_width=True)
            st.text("")
            st.text("")
            st.text("")
            st.text("")

        st.write("""
        ## Récapitulatif des performances des modèles   

        Ci-dessous, le tableau récapitulatif des performances de nos cinq modèles de régression selon les métriques évoqués: le score d'entraînement et de test (R²), ainsi que les erreurs MSE, MAE et RMSE :""") 
        # Création du DataFrame des metrics 
        data = {
        '': ['Score train', 'Score test  R²', 'MSE','MAE','RMSE'],
        'Linear Regression': [0.7995,0.7812,0.2648,0.3952,0.5146],
        'SVR': [0.8924,0.8634,0.1653,0.3107,0.4066],
        'Random Forest': [0.9858,0.8923,0.1304,0.2718,0.3611],
        'GBR': [0.9085, 0.8580, 0.1718,0.3212,0.4145],
        'XGB': [0.9990,0.8851,0.1391,0.2798,0.3729],
        }
        metrics = pd.DataFrame(data)
        st.dataframe(metrics)

        st.write("""Parmi nos modèles, le Random Forest a fourni le meilleur résultat, toutefois nous avons observé que ce modèle dépendait fortement d'une seule variable (PIB).Cette concentration excessive sur une variable peut 
                biaiser les résultats et limiter la capacité du modèle à prendre en compte d’autres facteurs. 
                \n En revanche, le Gradient Boosting Regressor s’est montré plus équilibré dans la feature importance, ce qui nous incite à privilégier ce modèle.
                \n
                \n Pour aller plus loin dans l'amélioration de ces algorithmes, c'est à dire en optimiser les hyperparamètres, nous avons retenu les modèles  **Gradient Boosting Regressor** et **XGBoost Regressor**, 
                plus complexes mais offrant un panel plus important d'optimisation de paramètres.
                \n
                \n Afin d'améliorer les deux modèles GBR et XGBoost, nous avons utilisé une recherche avec Gridsearch  en appliquant des valeurs sur les paramètres comme le learning_rate, max_depth, min_samples_leaf,
                  min_samples_split, n_estimators, subsample, colsample_bytree. Ces paramètres optimisés ont permis de renforcer la performance des modèles.""")

        X_train_copy = X_train_final.copy()
        y_train_copy = y_train.copy()
        X_test_copy = X_test_final.copy()
        y_test_copy = y_test.copy()
        

        # Ajuster le StandardScaler sur les données d'entraînement et l'enregistrer dans st.session_state (pour standardiser les valeurs données par visiteurs de l'app)
        scaler = StandardScaler().fit(X_train_final)
        st.session_state['scaler'] = scaler

        # Initialisation des modèles 
        gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

        # Gradient Boosting Regressor
        gbr.fit(X_train_copy, y_train_copy)
        y_pred_gbr = gbr.predict(X_test_copy)
        mse_gbr = mean_squared_error(y_test_copy, y_pred_gbr)
        r2_gbr = r2_score(y_test_copy, y_pred_gbr)


        # XGBoost Regressor
        xgb.fit(X_train_copy, y_train_copy)
        y_pred_xgb = xgb.predict(X_test_copy)
        mse_xgb = mean_squared_error(y_test_copy, y_pred_xgb)
        r2_xgb = r2_score(y_test_copy, y_pred_xgb)


        # les modèles pour les utiliser dans la prédiction
        st.session_state['gbr_model'] = gbr
        st.session_state['xgb_model'] = xgb

        #afficher les scores des modèles optimisés
        if st.checkbox("Afficher les scores des deux modèles optimisés"):
            # Création du DataFrame des résultats
            data2 = {
            '': ['Score train', 'Score test  R²', 'MSE','MAE','RMSE'],
            'GBR optimisé': [0.9851,0.8876,0.1360,0.2839,0.3688],
            'XGB optimisé': [0.9996, 0.8986, 0.1227,0.2602,0.3504]
                }
            metrics2 = pd.DataFrame(data2)
            st.dataframe(metrics2)


        
    elif st.session_state.selected_page == "Machine Learning":
        st.write("Sur cette page, vous pouvez explorer notre **algorithme de machine learning** en testant la précision de prédiction du score de bonheur des pays.") 
        st.write("Vous avez la possibilité d’utiliser des données issues du **World Happiness Report** pour vérifier la précision des prédictions, ou bien de tester avec des valeurs plus récentes, en vous basant sur des informations disponibles en ligne pour estimer un score de bonheur à venir.") 
        st.write("Pas de souci pour le format des données : notre algorithme standardisera automatiquement les valeurs saisies avant de lancer la prédiction.")
    
        if 'scaler' not in st.session_state:
            scaler = StandardScaler().fit(X_train_final)
            st.session_state['scaler'] = scaler 
        else:
            scaler = st.session_state['scaler'] 

        # Configuration des modèles
        def simulate_prediction(model, features):
            """Simule la prédiction pour les valeurs souhaitées en standardisant les entrées."""
            scaler = st.session_state['scaler']
            standardized_features = scaler.transform([features])
            prediction = model.predict(standardized_features)
            return prediction[0]

        # Variables explicatives et valeurs de l'exemple 2022 (France)
        feature_names = ['PIB par habitant', 'Soutien Social', 'Espérance de vie', 'Liberté de faire des choix de vie',
                     "Générosité", "Perceptions de la corruption", "Affect positif de l’année (0 si vous connaissez pas)",
                     "Affect négatif de l’année (0 si vous connaissez pas)", "Population", 
                     "Consommation d’énergie par habitant", "Appartenance à la région (1 ou 0) : Communauté des États indépendants",
                     "Appartenance à la région (1 ou 0) : Asie de l'Est", "Appartenance à la région (1 ou 0) : Amérique latine et Caraïbes", 
                     "Appartenance à la région (1 ou 0) : Moyen-Orient et Afrique", 
                     "Appartenance à la région (1 ou 0) : Amérique du Nord et ANZ (Australie et Nouvelle-Zélande)", 
                     "Appartenance à la région (1 ou 0) : Asie du Sud", "Appartenance à la région (1 ou 0) : Asie du Sud-Est",
                     "Appartenance à la région (1 ou 0) : Afrique subsaharienne", "Appartenance à la région (1 ou 0) : Europe de l’Ouest"]

        example_values = [1.863, 1.219, 0.808, 0.567, 0.07, 0.266, 0, 0, 64626624, 36051, 0, 0, 0, 0, 0, 0, 0, 0, 1]

        # Affichage de l'exemple 2022 et option de remplissage automatique
        st.info("Pour vous donner une idée, les valeurs pour la France en 2022 sont affichées ci-dessous. En cochant la case, vous accéderez à l'extrait des données issues des rapports Kaggle que nous avons utilisé. Appuyez sur le bouton ci-dessous pour remplir automatiquement les champs avec les données de cet exemple, ou bien saisissez vos propres valeurs pour simuler la prédiction du score de bonheur.")
        if st.checkbox("Afficher l'exemple 2022 (France)"):
            st.image("France 2022.png")

        if st.button("Remplir les champs avec les valeurs de l'exemple (2022 - France)"):
            for i, feature in enumerate(feature_names):
                st.session_state[f"gbr_{feature}"] = example_values[i]
                st.session_state[f"xgb_{feature}"] = example_values[i]

        # Création des onglets pour chaque modèle
        tabs = st.tabs(["Gradient Boosting Regressor", "XGBoost Regressor"])

        # Premier onglet - Gradient Boosting Regressor
        with tabs[0]:
            st.subheader("Simulation avec Gradient Boosting Regressor")
            user_inputs_gbr = []
            for feature in feature_names:
                value = st.session_state.get(f"gbr_{feature}", 0.0)
                input_value = st.number_input(f"Saisir la valeur de {feature} :", value=value, key=f"gbr_{feature}")
                user_inputs_gbr.append(input_value)

            if 'gbr_model' in st.session_state:
                if st.button("Simuler la prédiction (GBR)"):
                    gbr_prediction = simulate_prediction(st.session_state['gbr_model'], user_inputs_gbr)
                    st.write(f"**Score de bonheur prédit avec GBR :** {gbr_prediction:.2f}")
            else:
                st.write("GBR Non disponible")

        # Deuxième onglet - XGBoost Regressor
        with tabs[1]:
            st.subheader("Simulation avec XGBoost Regressor")
            user_inputs_xgb = []
            for feature in feature_names:
                
                value = st.session_state.get(f"xgb_{feature}", 0.0)
                input_value = st.number_input(f"Saisir la valeur de {feature} :", value=value, key=f"xgb_{feature}")
                user_inputs_xgb.append(input_value)

            if 'xgb_model' in st.session_state:
                if st.button("Simuler la prédiction (XGB)"):
                    xgb_prediction = simulate_prediction(st.session_state['xgb_model'], user_inputs_xgb)
                    st.write(f"**Score de bonheur prédit avec XGB :** {xgb_prediction:.2f}")
            else:
                st.write("XGB Non disponible")



    elif st.session_state.selected_page == "Conclusions et Perspectives":
        st.write("### Résumé des résultats obtenus et perspectives futures.")
        st.write("""
                 Après ajustement des hyperparamètres, les deux modèles optimisés — **Gradient Boosting Regressor (GBR)** et **XGBoost Regressor (XGB)** — ont démontré des performances élevées, notamment en termes de généralisation. 
                 
                 **Le modèle XGBoost**, en particulier, a obtenu des résultats légèrement supérieurs avec **des erreurs moyennes réduites** et **une meilleure explication de la variabilité de la variable cible**. 
                 De plus, XGBoost a révélé **une plus grande sensibilité aux indicateurs géographiques**, notamment en identifiant certaines régions, comme l’Amérique Latine et l’Afrique Subsaharienne, comme des facteurs influents dans le score de bonheur. 
                 
                 Cela constitue un résultat surprenant, suggérant que l’influence géographique, au-delà des indicateurs économiques classiques comme le PIB, peut jouer un rôle significatif dans les perceptions de bien-être.
                 
                 Ce constat soulève plusieurs hypothèses pour de futures recherches. 
                 L’impact élevé de certaines régions pourrait, par exemple, être lié à des facteurs culturels, sociaux ou même environnementaux propres à ces zones, qui ne sont pas explicitement intégrés dans notre jeu de données.""")

        st.write( "### Difficultés Scientifiques et Techniques Rencontrés")
        st.write("""
                La sélection et la gestion des variables explicatives ont constitué le principal défi de ce projet. 
                 
                Bien que l’ensemble de données comporte diverses variables économiques et sociales, certaines d’entre elles, comme le PIB, présentaient **une forte corrélation** avec d'autres indicateurs. Cette dépendance a compliqué la mise en place d’un modèle équilibré, capable de capturer des informations sans se concentrer excessivement sur une seule variable.
                Nous avons dû également introduire d'autres variables pour enrichir les données.
                
                Certaines tâches, en particulier **l’optimisation des hyperparamètres** pour les modèles GBR et XGBoost, ont nécessité davantage de temps que prévu. 
                Les tests d’optimisation via **GridSearchCV** ont impliqué plusieurs itérations longues, et le modèle GBR a notamment pris plus d’une heure à charger pour chaque ensemble d’hyperparamètres.
                                                 
                Les **scores finaux (R² proches de 0,89 et RMSE autour de 0,35)** démontrent une capacité de prédiction solide. Comparés aux benchmarks des modèles de base, tels que la régression linéaire, GBR et XGBoost offrent une généralisation plus performante. Les objectifs ont ainsi été atteints avec **un modèle de prédiction fiable**, exploitable pour des études socio-économiques sur la qualité de vie. 
                Ces résultats peuvent être intégrés dans des processus métiers liés à la prévision de bien-être, tels que des analyses pour des projets de développement, des études de politique publique, ou des indices de qualité de vie, en utilisant les variables macroéconomiques et sociales influentes identifiées.""")
        
        #afficher les scores des modèles optimisés
        if st.checkbox("Afficher les scores des deux modèles optimisés"):
            # Création du DataFrame des résultats
            data2 = {
            '': ['Score train', 'Score test  R²', 'MSE','MAE','RMSE'],
            'GBR optimisé': [0.9851,0.8876,0.1360,0.2839,0.3688],
            'XGB optimisé': [0.9996, 0.8986, 0.1227,0.2602,0.3504]
                }
            metrics2 = pd.DataFrame(data2)
            st.dataframe(metrics2)

        st.write(" ### Pistes d'amélioration")
        st.write(" Pour améliorer la performance des modèles, plusieurs pistes peuvent être envisagées :") 
        st.write("""
                - **Augmentation des données** : Intégrer davantage de données ou de sources externes pour limiter la variance des prédictions et enrichir les caractéristiques.      
                - **Assemblage de modèles** : Utiliser des techniques d'assemblage (stacking ou bagging) pour combiner les forces de GBR et XGBoost.        
                - **Optimisation avancée des hyperparamètres** : Envisager des techniques de recherche bayésienne pour une exploration plus fine des paramètres.     
                - **Ajout de variables explicatives** : Introduire des données sur les facteurs culturels, éducatifs, ou politiques pour enrichir les prédictions.
            """)

    elif st.session_state.selected_page == "Auteurs":

        # Auteur 1: Agnesa Kurusina
        st.subheader("Agnesa Kurusina")
        st.write("""
        D'origine en Commerce international, je suis actuellement Responsable du développement commercial dans un hôtel 4 étoiles à Biarritz. Mes missions principales consistent à optimiser le chiffre d'affaires de l'entreprise grâce à l'analyse de la performance et au benchmarking du marché, ainsi qu'à la gestion du portefeuille professionnel de l'établissement, notamment pour l'accueil de groupes et de séminaires. Afin d'améliorer mes performances, celles de mon entreprise, et de développer mes opportunités pour l'avenir, j'ai entrepris une formation de Data Analyst avec l'organisme Data Scientest. Mon objectif est d'apporter une approche axée sur les données pour résoudre des problématiques complexes et contribuer à des projets ambitieux, en alliant mes compétences techniques et métier pour orienter le business vers un modèle Data Driven. En parallèle, je m'investis dans des projets personnels liés à la langue japonaise, la cuisine et le blogging, dans un esprit d'amélioration continue et de développement personnel.""")
        st.markdown("[LinkedIn](https://www.linkedin.com/in/agnesa-kurusina-0a1804144)")

        # Auteur 2: Benjamin Giraud Matteoli
        st.subheader("Benjamin Giraud Matteoli")
        st.write("""
        Bonjour, je m'appelle Benjamin Giraud Matteoli. Fort de plus de 7 ans d'expérience en finance, audit et conseil, 
    j'accompagne les directions financières dans la modélisation, le reporting et la migration de systèmes d’information. 
    Ayant travaillé pour des groupes internationaux et maîtrisant les enjeux de post-intégration et de conformité, 
    j'ai décidé de développer de nouvelles compétences que j'estime élémentaires aujourd'hui en tant que Data Analyst 
    afin de compléter mes compétences métiers avec des skills plus techniques.
    """)
        st.markdown("[LinkedIn](https://www.linkedin.com/in/benjamin-giraud-matteoli-57942775/)")

        # Auteur 3: Marie Prévôt
        st.subheader("Marie Prévôt")
        st.write("""
        Titulaire d’un diplôme de master en Biologie, spécialité Neurosciences cellulaires et intégratives, j’ai pu étoffer mon profil au travers d’expériences professionnelles variées, que ce soit dans le domaine de l’écologie où j’ai pu mener à bien différent projets en tant que chargée de mission ou dans l’optique où mon travail consistait à accompagner le client dans la définition de son besoin et d’y répondre avec la solution technique la plus adaptée.
    N’étant pas tout à fait épanouie dans ce secteur je souhaite aujourd’hui me tourner vers la data, qui me semble plus en phase avec mon goût pour les chiffres et le sens de l’analyse.""")

        # Auteur 4: Nimol Mann
        st.subheader("Nimol Mann")
        st.write("""
        Fort de 20 ans d’expérience en supply chain approvisionnement et une appétence particulière pour la data, j'ai régulièrement manipulé des données (extractions datawarehouse, Access, Excel et BI) pour construire des analyses, tableaux de bord, KPIs et rapports.
    En reconversion dans le domaine de la data, je suis la formation Data Analyst chez Datascientest""")
        st.markdown("[LinkedIn](https://www.linkedin.com/in/nimol-mann-30a4a2112/)")

        

else:
    st.image("1690960666491.png")
    st.markdown(
        """
        <div style='text-align: center; font-size: 16px; line-height: 1.6;'>
            Ce projet vise à <strong>prédire le score de bonheur des pays</strong> à partir de diverses caractéristiques socio-économiques, 
            en s'appuyant sur des données historiques et des techniques de machine learning. <br><br> </div>""", unsafe_allow_html=True)

    st.markdown(
        """
        <div style='text-align: center; font-size: 16px; line-height: 1.6;'>
            En utilisant des indicateurs comme le <strong>PIB par habitant</strong>, <strong>le soutien social</strong>, <strong>l’espérance de vie</strong>, 
            <strong>la perception de la liberté</strong> et autres, le modèle tente d’estimer un <strong>score de bien-être national</strong>. <br><br> </div>""", unsafe_allow_html=True)

    st.markdown(
        """
        <div style='text-align: center; font-size: 16px; line-height: 1.6;'>
            La méthodologie comprend <strong>le prétraitement de données</strong> pour assurer leur qualité et cohérence, <strong>la standardisation des variables</strong>, 
            et <strong>l'entraînement de 5 modèles</strong>, puis ajustement des hyperparamètres pour trouver le meilleur algorithme de Machine Learning. <br><br>
            Le tableau de bord interactif offre <strong>une visualisation</strong> ainsi qu'une <strong>simulation dynamique de prédictions</strong>, 
            vous permettant d’explorer l'impact de chaque indicateur sur le bien-être des pays.<br><br> </div>""", unsafe_allow_html=True)

    st.markdown(
        """
        <div style='text-align: center; font-size: 18px; line-height: 1.6; color: green;'>
        <strong>Utilisez le menu de gauche pour naviguer entre les sections.</strong>
        </div>""", unsafe_allow_html=True)

