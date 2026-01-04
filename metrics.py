import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import dice_ml
from dice_ml.utils import helpers

def evaluate_dice_cf(query_instance, cf_list, model_predict_func, 
                      continuous_features, categorical_features, 
                      mad_dict, desired_class=1):
    """
    Implementierung der Evaluationsmetriken aus Kapitel 4.
    """
    k = len(cf_list)
    d = query_instance.shape[1]
    
    # 1. Gültigkeit (%ValidCFs) [cite: 169-172]
    # Zählt eindeutige CFs, die zur gewünschten Klasse führen
    unique_cfs = cf_list.drop_duplicates()
    # Annahme: Binäre Klassifikation (f(c) > 0.5 für Klasse 1) [cite: 170]
    valid_cfs = unique_cfs[model_predict_func(unique_cfs) > 0.5] if desired_class == 1 \
                else unique_cfs[model_predict_func(unique_cfs) <= 0.5]
    validity = len(valid_cfs) / k

    # Hilfsfunktionen für Distanzen
    def dist_cont(c1, c2):
        # L1-Distanz skaliert durch MAD [cite: 150, 174]
        diffs = []
        for feat in continuous_features:
            val = abs(c1[feat] - c2[feat]) / mad_dict[feat]
            diffs.append(val)
        return np.mean(diffs)

    def dist_cat(c1, c2):
        # 0/1-Funktion: Zählt Änderungen [cite: 149, 185]
        changes = sum(1 for feat in categorical_features if c1[feat] != c2[feat])
        return changes / len(categorical_features)

    # 2. Proximität (Nähe zur Eingabe) [cite: 173-196]
    cont_prox = -np.mean([dist_cont(cf, query_instance.iloc[0]) for _, cf in cf_list.iterrows()]) # [cite: 178]
    cat_prox = 1 - np.mean([dist_cat(cf, query_instance.iloc[0]) for _, cf in cf_list.iterrows()]) # [cite: 193]

    # 3. Sparsität (Anzahl unveränderter Merkmale) [cite: 197-205]
    total_changes = 0
    for _, cf in cf_list.iterrows():
        for col in cf_list.columns:
            if cf[col] != query_instance.iloc[0][col]:
                total_changes += 1
    sparsity = 1 - (total_changes / (k * d)) # [cite: 204]

    # 4. Diversität (Unterschiede zwischen CFs) [cite: 206-217]
    pairs = list(combinations(cf_list.iterrows(), 2))
    num_pairs = len(pairs)
    
    cont_div = sum(dist_cont(p1[1], p2[1]) for p1, p2 in pairs) / num_pairs # [cite: 211]
    cat_div = sum(dist_cat(p1[1], p2[1]) for p1, p2 in pairs) / num_pairs # [cite: 211]
    
    # Count Diversity [cite: 216]
    total_diff_features = 0
    for p1, p2 in pairs:
        total_diff_features += sum(1 for col in cf_list.columns if p1[1][col] != p2[1][col])
    count_div = total_diff_features / (num_pairs * d)

    return {
        "Validity": validity,
        "Continuous Proximity": cont_prox,
        "Categorical Proximity": cat_prox,
        "Sparsity": sparsity,
        "Continuous Diversity": cont_div,
        "Categorical Diversity": cat_div,
        "Count Diversity": count_div
    }

def calculate_mad(df, continuous_features):
    mad_dict = {}
    for feat in continuous_features:
        # Berechne den Median des Features
        median = df[feat].median()
        # Berechne die mittlere absolute Abweichung
        mad = (df[feat] - median).abs().median()
        
        # Sicherheits-Check: Falls MAD 0 ist (zu viele identische Werte), 
        # nimm eine kleine Konstante oder die Standardabweichung
        if mad == 0:
            mad = df[feat].std() if df[feat].std() != 0 else 1.0
            
        mad_dict[feat] = mad
    return mad_dict

if __name__ == "__main__":
    # 1. Datenvorbereitung (Eingabe aus Tabelle 5.2) [cite: 390]
    query_instance = pd.DataFrame([{
        "age": 26, "workclass": "Goverment", "education": "Some-college", 
        "marital status": "Single", "occupation": "Service", "race": "Other", 
        "gender": "Female", "h/week": 10
    }])

    # 2. CFs aus Tabelle 5.3 (Hardcoded) [cite: 399]
    cfs_hardcoded = pd.DataFrame([
        {"age": 26, "workclass": "Goverment", "education": "Masters", "marital status": "Married", "occupation": "Service", "race": "Other", "gender": "Female", "h/week": 90},
        {"age": 26, "workclass": "Goverment", "education": "Doctorate", "marital status": "Married", "occupation": "Service", "race": "Other", "gender": "Female", "h/week": 58},
        {"age": 88, "workclass": "Goverment", "education": "Some-college", "marital status": "Married", "occupation": "Service", "race": "Other", "gender": "Female", "h/week": 10},
        {"age": 26, "workclass": "Goverment", "education": "Bachelors", "marital status": "Married", "occupation": "Service", "race": "Other", "gender": "Male", "h/week": 72}
    ])

    dataset = helpers.load_adult_income_dataset()
    dataset = dataset.rename(columns={'hours_per_week': 'h/week', 'marital_status': 'marital status'})
    target = dataset["income"]
    train_dataset, test_dataset, y_train, y_test = train_test_split(dataset,
                                                                    target,
                                                                    test_size=0.2,
                                                                    random_state=0,
                                                                    stratify=target)
    # 3. Feature-Definitionen [cite: 316, 351]
    cont_feats = ["age", "h/week"]
    cat_feats = ["workclass", "education", "marital status", "occupation", "race", "gender"]
    
    # 4. Dummy-Vorhersagefunktion (Simuliert das ML-Modell)
    # Da alle CFs in Tabelle 5.3 als gültig (Einkommen=1) gelistet sind [cite: 399]
    def dummy_predict(df):
        return np.array([1.0] * len(df)) 

    # 5. MAD-Werte (Beispielwerte für Skalierung kontinuierlicher Features) [cite: 151]
    # mad_dict = {"age": 12.0, "h/week": 5.0}
    mad_dict = calculate_mad(train_dataset, cont_feats)

    # 6. Evaluation ausführen
    results = evaluate_dice_cf(
        query_instance=query_instance,
        cf_list=cfs_hardcoded,
        model_predict_func=dummy_predict,
        continuous_features=cont_feats,
        categorical_features=cat_feats,
        mad_dict=mad_dict,
        desired_class=1
    )

    # 7. Ergebnisse anzeigen
    print("--- Evaluationsergebnisse (Kapitel 4) ---")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    # Eingabevektor bleibt identisch (Tabelle 5.2) [cite: 390] ================================
    query_instance = pd.DataFrame([{
        "age": 26, "workclass": "Goverment", "education": "Some-college", 
        "marital status": "Single", "occupation": "Service", "race": "Other", 
        "gender": "Female", "h/week": 10
    }])

    # Neue CFs mit Einschränkungen aus Tabelle 5.4 
    # Nur erlaubte Merkmale wurden geändert: workclass, h/week, education, occupation, marital_status [cite: 412]
    cfs_constrained = pd.DataFrame([
        {"age": 26, "workclass": "Goverment", "education": "Doctorate", "marital status": "Married", "occupation": "Service", "race": "Other", "gender": "Female", "h/week": 37},
        {"age": 26, "workclass": "Goverment", "education": "Prof-school", "marital status": "Single", "occupation": "Professional", "race": "Other", "gender": "Female", "h/week": 37},
        {"age": 26, "workclass": "Goverment", "education": "Prof-school", "marital status": "Married", "occupation": "Service", "race": "Other", "gender": "Female", "h/week": 39},
        {"age": 26, "workclass": "Goverment", "education": "Doctorate", "marital status": "Married", "occupation": "Service", "race": "Other", "gender": "Female", "h/week": 45}
    ])

    results_constrained = evaluate_dice_cf(
        query_instance=query_instance,
        cf_list=cfs_constrained,
        model_predict_func=dummy_predict,
        continuous_features=["age", "h/week"],
        categorical_features=["workclass", "education", "marital status", "occupation", "race", "gender"],
        mad_dict={"age": 12.0, "h/week": 5.0},
        desired_class=1
    )

    print("--- Evaluation: CFs mit Einschränkungen (Kapitel 5.2.2) ---")
    for metric, value in results_constrained.items():
        print(f"{metric}: {value:.4f}")

    # 8. Analyse der Entscheidungsgrenze mittels 1-NN Proxy-Modell (Kapitel 5.3) =================

    def analyze_boundary(query_instance, cf_list):
        # 1. Daten kombinieren (Eingabe + CFs)
        # Die Eingabe hat Klasse 0, alle CFs haben Klasse 1
        X = pd.concat([query_instance, cf_list], ignore_index=True)
        y = np.array([0] + [1] * len(cf_list))
        
        # 2. One-Hot-Encoding für kategoriale Features
        # (Notwendig für die Distanzberechnung im 1-NN)
        cat_features = ["workclass", "education", "marital status", "occupation", "race", "gender"]
        X_encoded = pd.get_dummies(X, columns=cat_features)
        
        # 3. Training des 1-NN Proxy-Modells
        # Dieses Modell repräsentiert nun die "gelernte" Entscheidungsgrenze
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_encoded, y)
        
        # 4. Analyse der "Grenznähe"
        # Wir messen die Distanz von der Eingabe zum nächsten Punkt der Gegenklasse
        distances, indices = knn.kneighbors(X_encoded.iloc[0:1], n_neighbors=2)
        
        return {
            "boundary_distance": distances[0][1], # Distanz zum nächsten CF
            "nearest_cf_index": indices[0][1] - 1  # Welches CF definiert die Grenze?
        }

    # --- Durchführung für deine Beispiele ---

    # Ergebnisse für unbeschränkte CFs (Tabelle 5.3)
    res_unconstrained = analyze_boundary(query_instance, cfs_hardcoded)

    # Ergebnisse für beschränkte CFs (Tabelle 5.4)
    res_constrained = analyze_boundary(query_instance, cfs_constrained)
    print("--- Analyse der Entscheidungsgrenze mittels 1-NN Proxy-Modell ---")
    print(f"Unbeschränkt - Distanz zur Grenze: {res_unconstrained['boundary_distance']:.2f}")
    print(f"Beschränkt   - Distanz zur Grenze: {res_constrained['boundary_distance']:.2f}")