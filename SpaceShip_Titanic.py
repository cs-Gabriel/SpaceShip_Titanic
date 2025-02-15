#Importando Bibiliotecas
import pandas as pd

from sklearn.tree import DecisionTreeClassifier, plot_tree

import matplotlib.pyplot as plt

#%%
#Usando a função read_csv para ler os aquivos de teste e de treinamento
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#%% Exploração inicial
#Utilisa a funcão .head() para explorar o DataFrame
train.head()

#Utiliza função info() para observar o quantas observações tem em cada variavel
print(train.info())

#Ver se tem mt Nan
print(train.isnull().sum())
print('___________________')
print(test.isnull().sum())

#%%Tratar os null

# Substituir valores NaN ao invés de excluir todas as linhas
#Tratando Treino
train.fillna({
    'Age': train['Age'].median(),
    'CryoSleep': False,
    'VIP': False,
    'RoomService': 0,
    'FoodCourt': 0,
    'ShoppingMall': 0,
    'Spa': 0,
    'VRDeck': 0
}, inplace=True)

#tratando teste
test.fillna({
    'Age': test['Age'].median(),
    'CryoSleep': False,
    'VIP': False,
    'RoomService': 0,
    'FoodCourt': 0,
    'ShoppingMall': 0,
    'Spa': 0,
    'VRDeck': 0
}, inplace=True)

#%% Categorizando a idade
train['cat_idade'] = pd.cut(train['Age'], bins=[0, 18, 30, 40, 50, 80], labels=[0 ,1, 2, 3, 4])

#Covertendo Booleanos para int
train['CryoSleep'] = train['CryoSleep'].astype(int)
train['VIP'] = train['VIP'].astype(int)

#excluindo colunas que não ajudam
train.drop(columns = ['VRDeck', 'Spa', 'ShoppingMall', 'FoodCourt', 'RoomService'], inplace= True)

#%% Definino as variaveis para treinar a arvore
X_train = train[['cat_idade', 'CryoSleep', 'VIP']]
Y_train = train['Transported'].astype(int)

#%%Iniciando a arvore

# Criando o modelo de árvore de decisão
arvore = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)

# Treinando o modelo com os dados de treino
arvore.fit(X_train, Y_train)

# Visualizando a árvore gerada
plt.figure(figsize=(20, 10))
plot_tree(
    arvore,
    feature_names=X_train.columns.tolist(),
    class_names=["Não Transportado", "Transportado"],
    filled=True
)
plt.show()

#%% Testando a Arvore

#Categorizando a idade na base de teste
test['cat_idade'] = pd.cut(test['Age'], bins=[0, 18, 30, 40, 50, 80], labels=[0 ,1, 2, 3, 4])

#Somando todos os gastos dos passageiros 'Essa variavel acabei não utilizando, mas futuramente vou avalia, por isso deichei aqui'
test["total_gasto"] = test['RoomService'] + test['FoodCourt'] + test['ShoppingMall'] + test['Spa'] + test['VRDeck']


#Covertendo Booleanos para int
test['CryoSleep'] = test['CryoSleep'].astype(int)
test['VIP'] = test['VIP'].astype(int)

#%% Fazendo a previsão
#definindo as variaveis
X_teste = test[['cat_idade', 'CryoSleep', 'VIP']]

#Executando a previsão
Y_pred = arvore.predict(X_teste)

#%% Adicionando PassengerId ao DataFrame de resultados
resultado = pd.DataFrame({
    'PassengerId': test['PassengerId'],  # Identificador do passageiro
    'Transported': pd.Series(Y_pred).replace({0: False, 1: True})  # Resultado da previsão
})

# Exibir as primeiras linhas para conferir
print(resultado.head())

# Salvar como CSV para análise posterior
resultado.to_csv('previsoes_passageiros.csv', index=False)

