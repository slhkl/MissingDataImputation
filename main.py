import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Örnek veri oluşturalım
data = {
    'Id': [1, 2, 3, 4, 5],
    'Color': [5, 4, None, 3, 2],  # Örnek eksik veri
    'Shape': [3, None, 5, None, 2],  # Örnek eksik veri
    'Size': [4, 3, 5, None, 1],  # Örnek eksik veri
}

df = pd.DataFrame(data)

# Eksik değerleri doldurmak için bir karar ağacı regresyon modeli oluşturalım
for column in df.columns:
    if df[column].isnull().any():
        # Eğitim veri kümesi: eksik olmayan değerler
        train_data = df.dropna(subset=[column])
        X_train = train_data.drop(columns=['Id', column])
        y_train = train_data[column]

        # Test veri kümesi: eksik değerlerin olduğu satırlar
        test_data = df[df[column].isnull()]
        X_test = test_data.drop(columns=['Id', column])

        # Karar ağacı modelini oluştur ve eğit
        dt_model = DecisionTreeRegressor()
        dt_model.fit(X_train, y_train)

        # Eğitilen modeli kullanarak eksik değerleri tahmin et
        y_pred = dt_model.predict(X_test)

        # Tahmin edilen değerleri eksik değerlerin olduğu yerlere yerleştir
        df.loc[test_data.index, column] = y_pred

print("Eksik veriler doldurulduktan sonra veri özeti:")
print(df)
