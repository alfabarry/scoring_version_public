import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt




hearder_part =st.container()
proba_part=st.container()
side_part = st.container()
mean_part = st.container()
graphe_part = st.container()

titre =""" <div style="background-color: tomato; padding=20px; text-align:center">
<h1> Définir la chance qu'un emprunt soit remboursé </h1> </div> <br/><br/>
"""
with hearder_part:
    st.markdown(titre, unsafe_allow_html=True)

    ident = st.number_input("Saisir l'identifiant du client", min_value=000000)

    b1, b2 = st.columns([2, 2])
    butt1 = b1.button("Afficher les enregistrements du client", key='enregistrement')
    butt2 =b2.button("Afficher la probabilité du remboursement", key='probabilite')


@st.cache
def data_imp():
    data= pd.read_csv("application_train.csv", index_col="SK_ID_CURR")
    return data

data = data_imp()

dstat = data[["AMT_INCOME_TOTAL", "AMT_CREDIT", "DAYS_BIRTH", "AMT_ANNUITY", "TARGET", "CODE_GENDER", "NAME_FAMILY_STATUS", "ORGANIZATION_TYPE", "NAME_INCOME_TYPE",
             "NAME_EDUCATION_TYPE"]]


@st.cache
def data_proba():
    table_proba = pd.read_csv("table_proba.csv", index_col="SK_ID_CURR")
    return table_proba

@st.cache
def dtest ():
    data_test = pd.read_csv("data_test.csv", index_col="SK_ID_CURR")
    return data_test


data_test = dtest()
table_proba = data_proba()

#if ((ident not in data_test.index) and butt1) or ((ident not in data_test.index) and butt2) or ((ident not in data_test.index) and butt3):
 #   st.warning("L'identifiant n'existe pas")
#else :

with proba_part:
    st.markdown("# *** ")

    if ((ident not in data_test.index) and butt1):
        st.warning("L'identifiant n'existe pas")
    else :
        if (ident in table_proba.index) and butt1:
            st.write (data_test.loc[data_test.index==ident])



    if ((ident not in data_test.index) and butt2):
        st.warning("L'identifiant n'existe pas")
    else :

        if (ident in data_test.index) and butt2:

           # Charger le modèle

            p7_pipeline_modele_scoring = joblib.load("p7_pipeline_implementez_modele_scoring.sav")

            # definir instance
            instance_to_predict = data_test.loc[[ident]]
            proba_predite = p7_pipeline_modele_scoring.predict_proba(instance_to_predict)[0,1]


            st.markdown("### La probabilité de défaillance prédite pour ce client est : ")
            st.write(proba_predite)


            if (proba_predite <= 0.5) & (proba_predite >= 0.3):
                st.markdown("### Il y a une chance que le crédit soit remboursé")

            elif (proba_predite < 0.3):
                st.markdown("### Il y a une fort chance que le crédit soit remboursé")

            else :
                st.markdown ("### Attention : il y a un fort risque à faire du crédit à ce client ")


with side_part:
    st.sidebar.header("Choisir des options d'analyse")
    option = st.sidebar.selectbox("",
                                  ["", "Par classe", "Par status marital", "Par source de revenu",
                                   "Par type d'entreprise","Par niveau d'éducation"])

    st.sidebar.markdown("#### Les options d'analyse permettent de voir la répartition des "
                "emprunteurs selon differents critères ")


st.cache(allow_output_mutation=True)
with graphe_part:
    st.markdown("# *** ")
    data0 = dstat[dstat["TARGET"] == 0]
    data1 = dstat[dstat["TARGET"] == 1]


    if option == "":
        with mean_part:
            st.markdown("# *** ")

            st.markdown(" ## Dispersion des revenus des clients ")

            X0 =np.log(data0["AMT_INCOME_TOTAL"].dropna())
            X1 = np.log(data1["AMT_INCOME_TOTAL"].dropna())

            fig, ax = plt.subplots()
            green_diamond = dict(markerfacecolor='g', marker='D')
            ax.set_title("Le logarithme des revenus")
            ax.boxplot([X0, X1], flierprops=green_diamond, vert=False, labels=["Non défaillant ", "Défaillant"] ) #, bins=20, color=['g', 'b'])
            ax.set_ylabel("Classe d'appartenance")
            st.pyplot(fig)



            st.title("Caracteristique centrale: moyenne")

            revenu_mean = (dstat["AMT_INCOME_TOTAL"].dropna()).mean()
            credit_mean = (dstat["AMT_CREDIT"].dropna()).mean()
            age_mean = ((dstat["DAYS_BIRTH"].dropna()) / -365).mean()
            remb_mean = (dstat["AMT_ANNUITY"].dropna()).mean()

            st.write("Le revenu moyen des emprunteurs:", int(revenu_mean))
            st.write("L'âge moyen des emprunteurs:", int(age_mean))
            st.write("La somme moyenne des crédits:", int(credit_mean))
            st.write("La somme moyenne des remboursements:", int(remb_mean))

            st.write("Les emprunteurs de la banque ont en moyenne", int(age_mean),
                     "ans et dispose d'un revenu moyen de", int(revenu_mean), " unités monétaires. "
                                                                              "Ils empruntent en moyenne",
                     int(credit_mean), " qu'ils remborsent en moyenne de", int(remb_mean), "par an")


            st.title("Caracteristique centrale: mediane")

            revenu_median = (dstat["AMT_INCOME_TOTAL"].dropna()).median()
            credit_median = (dstat["AMT_CREDIT"].dropna()).median()
            age_median = ((dstat["DAYS_BIRTH"].dropna()) / -365).median()
            remb_median = (dstat["AMT_ANNUITY"].dropna()).median()

            st.write("Le revenu median des emprunteurs:", int(revenu_median))
            st.write("L'âge median des emprunteurs:", int(age_median))
            st.write("La somme médiane des crédits:", int(credit_median))
            st.write("La somme médiane des remboursements:", int(remb_median))

            st.write("La moitié des emprunteurs de la banque sont âgés au plus de", int(age_median), "ans. Elle a un revenu au plus de", int(revenu_median), " unités monétaires. "
                        "La moitié ont empuntés au plus ",int(credit_median), " et remboursent au plus ", int(remb_median), " par an")



    if option == "Par classe":

        st.markdown("## Répartition des emprunteurs par classe \n {0: bon, 1: mauvais}")
        st.bar_chart(data["TARGET"].value_counts())

        st.markdown("## Répartition des bons emprunteurs par genre")
        st.bar_chart(data0["CODE_GENDER"].value_counts())

        st.markdown("## Répartition des mauvais emprunteurs par genre")
        st.bar_chart(data1["CODE_GENDER"].value_counts())

    if option == "Par status marital":

        st.markdown("## Répartition des emprunteurs par status du mariage ")
        st.bar_chart(data["NAME_FAMILY_STATUS"].value_counts())

        st.markdown("## Répartition des bons emprunteurs par status du mariage")
        st.bar_chart(data0["NAME_FAMILY_STATUS"].value_counts())

        st.markdown("## Répartition des mauvais emprunteurs par status du mariage")
        st.bar_chart(data1["NAME_FAMILY_STATUS"].value_counts())


    if option == "Par type d'entreprise":

        st.markdown("## Répartition des emprunteurs par type d'entreprise ")
        st.bar_chart(data["ORGANIZATION_TYPE"].value_counts())

        st.markdown("## Répartition des bons emprunteurs par type d'entreprise")
        st.bar_chart(data0["ORGANIZATION_TYPE"].value_counts())

        st.markdown("## Répartition des mauvais emprunteurs par type d'entreprise")
        st.bar_chart(data1["ORGANIZATION_TYPE"].value_counts())


    if option == "Par source de revenu":
        st.markdown("## Répartition des emprunteurs par source de revenu")
        st.bar_chart(data["NAME_INCOME_TYPE"].value_counts())

        st.markdown("## Répartition des bons emprunteurs par source de revenu")
        st.bar_chart(data0["NAME_INCOME_TYPE"].value_counts())

        st.markdown("## Répartition des mauvais emprunteurs par source de revenu")
        st.bar_chart(data1["NAME_INCOME_TYPE"].value_counts())

    if option == "Par niveau d'éducation":
        st.markdown("## Répartition des emprunteurs par niveau d'éducation")
        st.bar_chart(data["NAME_EDUCATION_TYPE"].value_counts())

        st.markdown("## Répartition des bons emprunteurs par niveau d'éducation")
        st.bar_chart(data0["NAME_EDUCATION_TYPE"].value_counts())

        st.markdown("## Répartition des mauvais emprunteurs par niveau d'éducation")
        st.bar_chart(data1["NAME_EDUCATION_TYPE"].value_counts())
