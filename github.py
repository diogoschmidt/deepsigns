
# coding: utf-8

# In[ ]:


from math import sqrt
from numpy import concatenate
import pandas as pd
import numpy as np
import random
import collections
import copy, csv
from matplotlib import pyplot
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from datetime import datetime
from sklearn import preprocessing


# In[ ]:


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('at%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('at%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('at%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


# In[ ]:


#importação multinivel
dateparse = lambda dates: [pd.datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in dates]
dataset_raw = read_csv('chartevents_apache_02.csv', header=0, index_col=[0,1,2], quotechar='"', quoting=1, parse_dates=['charttime'], date_parser=dateparse)
#dataset_raw


# In[ ]:


# ----- Parâmetros: -----

##Versao
versao = '_V15'

## Variáveis: APACHE

## Granularidade da janela de tempo (split)
split = '30min'

## Limite do número de hadm_id para transformar Série Temporal
corte = 100

## Passo (Número de casos para treino, "hadm_id", a cada rodagem do treinamento)
passo = int(corte*0.7)   #Todos casos: split_tt

##  Número de casos para teste ("hadm_id")
casos_teste = int(corte*0.3)   #Todos casos: len(hadm_id) # 70%: int((passo/0.7)-passo)

## Número splits histórico para treino. Janela de Treino:
max_train = 48

## Quantidade de horas de predição
horas = 24

## Número splits de predição. Janela de Predição. (4horas -> 240min)
#max_pred = 48
max_pred = horas*60//(int(split.strip('min')))

## Variação de Risco APACHE - Diferença do início para fim da janela
apache_delta = 4

## Valor de Risco para APACHE
apache_alert = 24

## Número neuronios LSTM
neuronios = 50 

## Épocas
epocas = 10

## Debug -> 0 - oculta / 1 - mostra
debug = 0


# In[ ]:


#cria array de indices(hadm_id)
hadm_id = copy.copy(dataset_raw.index.levels[0].values)
#Randomiza lista de "hadm_id"
random.shuffle(hadm_id)
#Divide o numero TOTAL de hadm_id em 70/30
split_tt = int(len(hadm_id)*0.7)
#verifica se a variável "corte" de treino esta dentro do intervalo 70/30
if corte > split_tt:
    corte = split_tt
#aplica o corrte no array de indices(hadm_id)
hadm_id = hadm_id[0:corte]
#Redivide o numero de hadm_id ja cortado em 70/30
split_tt = int(len(hadm_id)*0.7)
#Verficia se a variavel "casos_teste"não execede o intervalo 70/30
if casos_teste > (len(hadm_id) - split_tt):
    casos_teste = (len(hadm_id) - split_tt)

print('Total de hadm_id:',len(hadm_id), 'Treino:', split_tt, 'Teste:', casos_teste)


# In[ ]:


#### Calculo APACHE ####
def apache(data_apache):
    pontos = 0

    #idade
    if  data_apache['idade'] < 45.0:
        pontos+=0
    else:
        if data_apache['idade'] < 55.0:
            pontos+=2
        else:
            if data_apache['idade'] < 65.0:
                pontos+=3
            else:
                if data_apache['idade'] < 75.0:
                    pontos+=5
                else:
                    pontos+=6    
    #temperartura
    if  data_apache['temp']  <= 85.9:
        pontos+=4
    else:
        if data_apache['temp']  <= 89.5:
            pontos+=3
        else:
            if data_apache['temp']  <= 93.1:
                pontos+=2
            else:
                if data_apache['temp']  <= 96.7:
                    pontos+=1
                else:
                    if data_apache['temp']  <= 101.2:
                        pontos+=0
                    else:
                        if data_apache['temp']  <= 102.1:
                            pontos+=1
                        else:
                            if data_apache['temp']  <= 105.7:
                                pontos+=3
                            else:
                                pontos+=4
    #pressão    
    if  data_apache['mpress'] <= 49.0:
        pontos+=4
    else:
        if data_apache['mpress'] <= 69.0:
            pontos+=2
        else:
            if data_apache['mpress'] <= 109.0:
                pontos+=0
            else:
                if data_apache['mpress'] <= 129.0:
                    pontos+=2
                else:
                    if data_apache['mpress'] <= 159.0:
                        pontos+=3
                    else:
                        pontos+=4     
    #frequencia
    if  data_apache['freq']  <= 39.0:
        pontos+=4
    else:
        if data_apache['freq']  <= 54.0:
            pontos+=3
        else:
            if data_apache['freq']  <= 69.0:
                pontos+=2
            else:
                if data_apache['freq']  <= 109.0:
                    pontos+=0
                else:
                    if data_apache['freq']  <= 139.0:
                        pontos+=2
                    else:
                        if data_apache['freq']  <= 179.0:
                            pontos+=3
                        else:
                            pontos+=4
    #respiracao
    if  data_apache['rr']  <= 5.0:
        pontos+= 4
    else:
        if data_apache['rr']  <= 9.0:
            pontos+=2
        else:
            if data_apache['rr']  <= 11.0:
                pontos+=1
            else:
                if data_apache['rr']  <= 24.0:
                    pontos+= 0
                else:
                    if data_apache['rr']  <= 34.0:
                        pontos+= 1
                    else:
                        if data_apache['rr']  <= 49.0:
                            pontos+= 3
                        else:
                            pontos+= 4
    # fio2/PaO2/AaDO2
    PaO2 = 0
    if  data_apache['pao2'] <= 54.0:
        PaO2+= 4
    else:
        if data_apache['pao2'] <= 60.0:
            PaO2+= 3
        else:
            if data_apache['pao2'] <= 70.0:
                PaO2+= 1
            else:
                PaO2+= 0

    gradiente = ((760-47)*data_apache['fio2'])-(data_apache['paCO2']/1)-data_apache['pao2']
    AaDO2 = 0
    if  gradiente <= 200.0:
        AaDO2+= 0
    else:
        if gradiente <= 349.0:
            AaDO2+= 2
        else:
            if gradiente <= 499.0:
                AaDO2+= 3
            else:
                AaDO2+= 4

    if data_apache ['pao2'] >= 0.5:
        pontos+=AaDO2
    else:
        pontos+=PaO2

    # ph     
    if  data_apache['ph']  <= 7.14:
        pontos+=4
    else:
        if data_apache['ph']  <= 7.24:
            pontos+=3
        else:
            if data_apache['ph']  <= 7.32:
                pontos+=2
            else:
                if data_apache['ph']  <= 7.49:
                    pontos+=0
                else:
                    if data_apache['ph']  <= 7.59:
                        pontos+=1
                    else:
                        if data_apache['ph']  <= 7.69:
                            pontos+=3
                        else:
                            pontos+= 4
    # sodio   
    if  data_apache['sodio'] <= 110.0:
        pontos+=4
    else:
        if data_apache['sodio'] <= 119.0:
            pontos+= 3
        else:
            if data_apache['sodio'] <= 129.0:
                pontos+= 2
            else:
                if data_apache['sodio'] <= 139.0:
                    pontos+=0
                else:
                    if data_apache['sodio'] <= 154.0:
                        pontos+= 1
                    else:
                        if data_apache['sodio'] <= 159.0:
                            pontos+=2
                        else:
                            if data_apache['sodio'] <= 179.0:
                                pontos+=3
                            else:
                                pontos+=4

    #potassio 
    if  data_apache['potasio'] <= 2.4:
        pontos+=4
    else:
        if data_apache['potasio'] <= 2.9:
            pontos+=2
        else:
            if data_apache['potasio'] <= 3.4:
                pontos+=1
            else:
                if data_apache['potasio'] <= 5.4:
                    pontos+=0
                else:
                    if data_apache['potasio'] <= 5.9:
                        pontos+=1
                    else:
                        if data_apache['potasio'] <= 6.9:
                            pontos+=3
                        else:
                            pontos+= 4
    # creatinina
    if  data_apache['creat'] <= 0.6:
        pontos+=2
    else:
        if data_apache['creat'] <= 1.4:
            pontos+=0
        else:
            if data_apache['creat'] <= 1.9:
                pontos+=2
            else:
                if data_apache['creat'] <= 3.4:
                    pontos+=3
                else:
                    pontos+=4
    # hematocrito    
    if  data_apache['hemat'] <= 20.0:
        pontos+=4
    else:
        if data_apache['hemat'] <= 29.9:
            pontos+=2
        else:
            if data_apache['hemat'] <= 45.9:
                pontos+=0
            else:
                if data_apache['hemat'] <= 49.9:
                    pontos+=1
                else:
                    if data_apache['hemat'] <= 59.9:
                        pontos+=2
                    else:
                        pontos+=4
    # leucocitos 
    if  data_apache['leuc'] <= 1.0:
        pontos+=4
    else:
        if data_apache['leuc'] <= 2.9:
            pontos+=2
        else:
            if data_apache['leuc'] <= 14.9:
                pontos+=0
            else:
                if data_apache['leuc'] <= 19.9:
                    pontos+=1
                else:
                    if data_apache['leuc'] <= 39.9:
                        pontos+=2
                    else:
                        pontos+= 4
    #coma
    pontos+=15-data_apache['coma']
    
    return pontos


# In[ ]:


### Função para Conversão dos dados em Dataframe de Série Temporal para cada "hadm_id" ####

#### Converte um conjunto de dados de uma admissão (hadm_id) em um dataframe de serie temporal ###
def serie_temp(dataset):
    #print('------------------DATASET--------------------')
    #print(dataset)
    #Transforma index idade em coluna
    dataset.reset_index(drop = False, inplace = True)
    dataset.set_index('charttime', inplace=True)
    #Ajusta a primeira hora da serie para a hora da primeira medida existente
    min_ini = dataset.index[0].strftime('%M') + 'min'
    #print('dataset.index[0]', dataset.index[0])
    #print('dataset.index[-1]', dataset.index[-1])
    #print('min_ini', min_ini)
    inicio = dataset.index[0] - pd.Timedelta(min_ini)
    #print('inicio', inicio)
    #cria um Dataframe para a Serie Final com um intervalo determinado
    df = pd.DataFrame
    #print(df)
    range_index = pd.date_range(inicio, dataset.index[-1], freq=split)
    #print(' - range_index', range_index)
    df = pd.DataFrame(index = range_index)
    #print(df)
    # Seleciona os atributos e reorganiza (média) dentro do intervalo defino para DF final
    #temperatura
    temp = dataset[dataset['itemid'] == 678]
    temp = temp.resample(split).mean()
    #Mean press
    mpress = dataset[dataset['itemid'] == 456]
    mpress = mpress.resample(split).mean()
    #frequencia
    freq = dataset[dataset['itemid'] == 211]
    freq = freq.resample(split).mean()
    #Respiratory Rate
    rr = dataset[dataset['itemid'] == 618]
    rr = rr.resample(split).mean()
    #FiO2
    fio2 = dataset[dataset['itemid'] == 190]
    fio2 = fio2.resample(split).mean()
    #PaO2
    pao2 = dataset[dataset['itemid'] == 779]
    pao2 = pao2.resample(split).mean()
    #paCO2
    paCO2 = dataset[dataset['itemid'] == 778]
    paCO2 = paCO2.resample(split).mean()
    #ph
    ph = dataset[dataset['itemid'] == 780]
    ph = ph.resample(split).mean()
    #sodio
    sodio = dataset[dataset['itemid'] == 837]
    sodio = sodio.resample(split).mean()
    #potasio
    potasio = dataset[dataset['itemid'] == 829]
    potasio = potasio.resample(split).mean()
    #Creatinine
    creat = dataset[dataset['itemid'] == 791]
    creat = creat.resample(split).mean()
    #hematrocito
    hemat = dataset[dataset['itemid'] == 813]
    hemat = hemat.resample(split).mean()
    #leucocito
    leuc = dataset[dataset['itemid'] == 861]
    leuc = leuc.resample(split).mean()
    #GCS Total
    coma = dataset[dataset['itemid'] == 198]
    coma = coma.resample(split).mean()
    
    
    #monta o dataframe temporal
    df['temp'] = temp['valuenum']       #1
    df['mpress'] = mpress['valuenum']   #2
    df['freq'] = freq['valuenum']       #3
    df['rr'] = rr['valuenum']           #4
    df['fio2'] = fio2['valuenum']       #5
    df['pao2'] = pao2['valuenum']       #6
    df['paCO2'] = paCO2['valuenum']     #7
    df['ph'] = ph['valuenum']           #8
    df['sodio'] = sodio['valuenum']     #9
    df['potasio'] = potasio['valuenum'] #10
    df['creat'] = creat['valuenum']     #11
    df['hemat'] = hemat['valuenum']     #12
    df['leuc'] = leuc['valuenum']       #13
    df['coma'] = coma['valuenum']       #14
    df['idade'] = dataset.iloc[0][0]    #15
    #Ultima coluna deve ser o indicie prognostico de comparaçao
    df['apache'] = 0                    #16

    #Preenche as lagunas de tempo repetindo ultimo valor
    df = df.fillna(method='pad')

    #completa valores de referencia para medidas ausentes
    references = {'temp': 98.96, 'mpress': 89.5, 'freq': 89.5, 'rr': 18, 'fio2': 0.21, 'pao2': 90, 'paCO2': 40, 'ph': 7.395, 'sodio': 139.5, 'potasio': 4.45, 'creat': 1, 'hemat': 37.95, 'leuc': 8.95, 'coma': 15}
    #references = 0
    df = df.fillna(value=references)
    #print('df.describe() DEPOIS de preenchimento de dados faltantes')
    #print(df.describe())
    #print(df.head(10))
    
    values = df.values
    # ensure all data is float
    values = values.astype('float32')
    
    #calculo do apache - calcula e armazena na Ultima coluna do DF
    for k in range(values.shape[0]):
        apache_calc = df.iloc[k][0:]
        values[k][-1] = apache(apache_calc)
        
    #normalize features
    #scaler = MinMaxScaler(feature_range=(0, 1))
    
    #scaled = scaler.fit_transform(values) 


    
######  values ou scaled ######
    # frame as supervised learning
    reframed = series_to_supervised(values, 1, 1)  #devolve um dataframe com o dobro de colunas "t-1" e "t"
    #print reframed.head
    
    if 'list' in globals():
        del list    
        
   
    return reframed


# In[ ]:


#### Cria um DF gigante com todos os dados ja em formato de serie temporal  ####  Muito demorado
#Cria um DF vazio com indice "hadm_id"
df_ts = pd.DataFrame(index = hadm_id)
df_ts['serie'] = ""
#insere os dados ja montados em cada hadm_id
for x in range(0,len(hadm_id),1):
    df_ts.at[hadm_id[x],'serie'] = serie_temp(dataset_raw.loc[hadm_id[x]])
#df_ts.head()


# In[ ]:


#apaga variáveis para rodar novamente o treinamento do zero
if 'model' in vars() or 'model' in globals():
    del model, serie, ini, fim#, pesos, pesos2

if 'ini' in vars() or 'ini' in globals():
    del ini, fim

if 'calc_atrib' in vars() or 'calc_atrib' in globals():
    del calc_atrib

if 'calc_indice' in vars() or 'calc_indice' in globals():
    del calc_indice


# In[ ]:


### Rodagem do modelo - TREINO e TESTE

#variável para guardar o erro de cada atributo
error_atrib = []
#variável para contar o volume de dados treinado
cont_reg = 0
#variável para guardar as predições de todos atributos
#calc_indice = []

#Calcula quantos casos serão usadados para rodar o treinamento - ini ao fim, caso a rodagem seja feita por lotes ("passo") sem zerar a configuração de rede 
if 'ini' not in vars() or 'ini' not in globals():
    ini = 0
    fim = passo
else: 
    ini = ini + passo
    fim = ini + passo
if (fim >= split_tt):
    fim = split_tt
if (ini >= split_tt):
    ini = split_tt

# Loop de todos atributos a serem preditos ***** exceto idade e APACHE (dois ultimos -2)****
for k in range(df_ts.loc[hadm_id[0]][0].shape[1]//2,df_ts.loc[hadm_id[0]][0].shape[1]-2):
    lista_nova = list(range(df_ts.loc[hadm_id[0]][0].shape[1]//2,df_ts.loc[hadm_id[0]][0].shape[1]))
    del lista_nova[k-df_ts.loc[hadm_id[0]][0].shape[1]//2]
    if (debug == 1):
        print('---------------')
        print('\n##### DEBUG:', lista_nova)
        print('---------------')

    #indicador de qual atributo será calculado
    atributo = k-(df_ts.loc[hadm_id[0]][0].shape[1]/2)+1
    if (debug == 1):
        print('\n======== TREINAMENTO ========')
        print('Atributo:', atributo)
        print('hadm_id range:',ini, 'a', fim)
        print('****************************\n')
    
############# TREINAMENTO #############    
    # Loop de hadm_id de treino
    if 'model' in vars() or 'model' in globals():
        del model
    #variável para contar o volume de dados treinado
    cont_reg = 0

    for x in range(ini,fim,1):
        if (debug == 1):
            print('\thadm_id[',x,']:',hadm_id[x])
       
        # Retira indice prognostico dos dados para treino (última coluna dos dados t-1) e as colunas dos atributos não preditos  (var_apach+lista_nova)******        
        var_apach = [((df_ts.loc[hadm_id[0]][0].shape[1]//2)-1)]
        serie = df_ts.loc[hadm_id[x]][0].drop(df_ts.loc[hadm_id[x]][0].columns[var_apach+lista_nova], axis=1)
        #print serie,'\n----'
        
        if serie.empty:
            if (debug == 1):
                print('\tDataFrame vazio!\n-------------------------------------\n\n')
        else:
            scaler = MinMaxScaler(feature_range=(0, 1))
            rescaled = scaler.fit_transform(serie)

            # split into train and test set
            #train = serie.values #old
            train = rescaled
            # split into input and outputs
            train_X, train_y = train[:, :-1], train[:, -1]
            # reshape input to be 3D [samples, timesteps, features]
            train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
            #print train_X
            # Monta a estrutura da rede
            if 'model' not in vars() or 'model' not in globals(): #nao cria novamente o modelo se for novo treinamento
                model = Sequential()
                model.add(LSTM(neuronios, input_shape=(train_X.shape[1], train_X.shape[2]), kernel_initializer='zeros'))
                model.add(Dense(1))
                model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])
                #print model.summary()

            #pesos = model.get_weights()
            if (debug == 1):
                print('\tRegistros:', serie.shape[0])
            # TREINAMENTO
            if (serie.shape[0] >= 10): #Utiliza somente dados de internações com mais de 10 intervalos de tempo
                history = model.fit(train_X, train_y, epochs=epocas, batch_size=72, verbose=0, shuffle=False, validation_split=0.1)
                #print 'Loss:', history.history['loss'][0], '->', history.history['loss'][-1]
                cont_reg = cont_reg + serie.shape[0]
                
                if (debug == 1):
                    # Gráfico da history for loss 
                    x = np.arange(0, epocas, 1)
                    y1 = history.history['loss']
                    y2 = history.history['val_loss']
                    y3 = history.history['acc']
                    y4 = history.history['val_acc']

                    fig, ax1 = plt.subplots()

                    ax2 = ax1.twinx()
                    ax1.plot(x, y1, 'g-', label='loss')
                    ax1.plot(x, y2, 'b-', label='val_loss')
                    ax2.plot(x, y3, 'r-', label='acc')
                    ax2.plot(x, y4, 'k-', label='val_acc')

                    ax1.set_xlabel('epoch')
                    ax1.set_ylabel('loss')
                    ax2.set_ylabel('acc')
                    ax1.legend(bbox_to_anchor=(1.45, 1))
                    ax2.legend(bbox_to_anchor=(1.45, 0.7))
                    pyplot.title('model loss and accuracy')
                    pyplot.show()
                
                
                
                if (debug == 1):
                    print('---------------------------------------------------\n')
                #scores = model.evaluate(test_X, test_y, verbose=1)  # Evaluate the trained model on the test set!
            #pesos2 = model.get_weights()
            #fim do for de treinamento
    if (debug == 1):
        print('======== FIM TREINO ========\n')
        print('\n##### DEBUG:\n', var_apach)
     
    
############# TESTE #############
    
    if (debug == 1):
        print('======== TESTE ========')
        print('Atributo:', atributo)
        print('hadm_id range:',split_tt, 'a', split_tt+casos_teste)
        print('****************************\n')
    erros = []
        
    # Loop de hadm_id de teste
    for y in range(split_tt,split_tt+casos_teste,1): # todos casos: len(hadm_id)        
        teste = df_ts.loc[hadm_id[y]][0].drop(df_ts.loc[hadm_id[y]][0].columns[var_apach+lista_nova], axis=1)
        
        #Guarda os dados de idade e apache para juntar ao DF no final
        var_idade_apache = list(range(0,df_ts.loc[hadm_id[0]][0].shape[1]))    
        del var_idade_apache[(df_ts.loc[hadm_id[0]][0].shape[1]//2)-1]
        del var_idade_apache[(df_ts.loc[hadm_id[0]][0].shape[1]//2)-2]
        dados_idade_apache = df_ts.loc[hadm_id[y]][0].drop(df_ts.loc[hadm_id[y]][0].columns[var_idade_apache], axis=1)
        dados_idade_apache = dados_idade_apache.values
        dados_idade_apache = dados_idade_apache[:max_pred,:]
        #print '\n##### DEBUG:\n', dados_idade_apache.shape
        #print '\n##### DEBUG:\n', teste.shape,'\n----'
        
        
        # verifica se tem dados suficiente para Janela de predição (no mínimo o numero de splits dentro da janela) e se são maior que zero
        if ((teste.shape[0] >= max_pred) and (teste.iloc[:,[0]].mean().values > 0)):
            # split into train and test sets
            test = scaler.fit_transform(teste)             
            #limita o Número de splits de predição 
            test =  test[:max_pred,:]
            
            # split into input and outputs
            test_X, test_y = test[:, :-1], test[:, -1]
            # reshape input to be 3D [samples, timesteps, features]
            test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
            
            # make a prediction
            yhat = model.predict(test_X)
            
            #print '\n##### DEBUG:\n', len(yhat),'\n----'
            
            test_X_final = test_X.reshape((test_X.shape[0], test_X.shape[2]))
            
            # iverte escala da prediçao - Uni a prediçao "yhat" com todos os dados para poder tranformar a escala
            inv_yhat = concatenate((test_X_final,yhat), axis=1)
            inv_yhat = scaler.inverse_transform(inv_yhat)
            # apos voltar a escala real deixa somente a prediçao
            inv_yhat = inv_yhat[:,-1]
            
            # iverte escala do TesteY (valores reais)
            inv_y = scaler.inverse_transform(test)
            inv_y = inv_y[:,-1]
            
            
            #print '\n##### DEBUG:\n', inv_yhat
                
            # Guarda as prediçoes na varável "calc_atrib"
            if 'calc_atrib' in vars() or 'calc_atrib' in globals():
                calc_atrib = np.append(calc_atrib,inv_yhat, axis=0)
                idade_apache = np.append(idade_apache,dados_idade_apache, axis=0)
            else:
                calc_atrib = np.array(inv_yhat)
                idade_apache = np.array(dados_idade_apache)
               
            
            # calcula o RMSE (inv_y->Real e inv_yhat->Predita)
            rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
            erros.append(rmse)

            if (debug == 1):
                #print ini, fim, 'hadm_id -> Test RMSE: %.3f' % rmse
                print('\thadm_id[',y,']:',hadm_id[y],'\n\tsplits:',teste.shape[0],'RMSE: %.3f' % rmse)
            
            if (debug == 1):
                #Grafico Predict x Real
                pyplot.plot(inv_y, label='Real')
                pyplot.plot(inv_yhat, label='Predict')
                pyplot.axis([0, max_pred, min(np.min(inv_y),np.min(inv_yhat))*0.99, max(np.max(inv_y),np.max(inv_yhat))*1.01])
                pyplot.title('Predict x Real')
                pyplot.ylabel('valor')
                pyplot.xlabel('splits')
                pyplot.legend()
                pyplot.show()
            
            
        else:
            if (debug == 1):
                print('\thadm_id[',y,']:',hadm_id[y],'\n\tSem dados suficente\n')
        if (debug == 1):
            print('----------------------------------------------------\n')
    if (debug == 1):
        print('======== FIM TESTE ========\n')

############# RESULTADOS #############
    # Resultado do RMSE de cada atributo
    print('======== RESULTADOS ========')
    print('Atributo:', atributo)
    s = pd.Series(erros)
    print('Casos de tese:',len(hadm_id) - split_tt, '- Testados:', s.shape[0])
    print('Mean squared error: %.2f' % s.mean())
    error_atrib.append(s.mean())

    # Guarda todas predições do atributo na variável "calc_indice"
    if 'calc_indice' in vars() or 'calc_indice' in globals():
        calc_indice[atributo] = calc_atrib
        #df_idade_apache[atributo] = idade_apache
    else:
        calc_indice = pd.DataFrame(data=calc_atrib, columns=[atributo])
        df_idade_apache = pd.DataFrame(data=idade_apache, columns=['15','16'])
    
    #limpa variavel das predições de cada atributo
    del calc_atrib, idade_apache
    
    print('======== FIM ========\n\n')
    print('***********************************************************************\n')


# In[ ]:


## Contabilização dos RESULTADOS
print('*** Configuração:***')
print('Split:', split, '/ Casos de treino:',passo,'/ Número de registros:', cont_reg, '/ Casos de teste:', casos_teste, '/ Testados:', s.shape[0], '/ Janela de Predição:',max_pred, '/ Neuronios:', neuronios, '/ Epocas:', epocas)

print('\n*** Resultados RMSE ATRIBUTOS - Etapa 1 ***')
#Contabilização do Erro médio de cada atributo
for w in range(0,len(error_atrib),1):
    print('Atributo ',w+1,': %.2f' % error_atrib[w])
soma_rmse_at = round(sum(error_atrib),2)
print('\nSoma RMSE:', soma_rmse_at)
med_rmse_at = round(sum(error_atrib)/len(error_atrib),2)
print('Média RMSE:', med_rmse_at)

#Cria um DF novo "result" com uma coluna para cada predição feita e renomeia as colunas para poder calcular o APACHE depois
result = pd.concat([calc_indice, df_idade_apache], axis=1)
result.columns = ['temp', 'mpress', 'freq', 'rr', 'fio2', 'pao2', 'paCO2', 'ph', 'sodio', 'potasio', 'creat', 'hemat', 'leuc', 'coma', 'idade', 'apache t-1']
#Calculo do APACHE com base nas varáveis preditas
apache_t = []
for h in range(result.shape[0]):
        apache_t_calc = result.iloc[h][0:]
        apache_t.append(apache(apache_t_calc))
        #print apache_t_calc, apache_t[h], '\n'
result['apache t'] = apache_t
#Calcula o ERRO ABSOLUTO entre o APACHE com as variáveis preditas e as reais
result['dif'] = abs(result['apache t']-result['apache t-1'])
#result

acertos = 0
for g in range (0,len(result['dif']),1):
    #print round(result['dif'][g],2),' - ',int(round(result['dif'][g],0))
    #if (int(round(result['dif'][g],0))==0):
    if (result['dif'][g] < 0.6):
        acertos = acertos+1

        
print('\n*** Resultados ÍNDICE  - Etapa 2 ***')
print('Predições:', (len(result['dif'])))
# Porcetagem de acertos do APACHE predito
acertos_indice = round(float(acertos)*100/(len(result['dif'])),2)
print('Acertos:', acertos_indice,'%')
# Erro Absoluto Média (MAE) das entre o APACHE predito e o real para cada predição (split)
e2_mae = round(result['dif'].describe()[1],2)
print('MAE:',e2_mae)
#RMSE APACHE real e APACHE predito
e2_rmse = round(sqrt(mean_squared_error(result['apache t-1'], result['apache t'])),2)
print('RMSE:', e2_rmse)


# In[ ]:


print('*** Utilidade - Etapa 3 ***\n')
#variável para registrar se houve algum alerta durante a janela
alert_true = 0
utilidade = result['apache t-1']
utilidade = utilidade.values.reshape(utilidade.shape[0]//max_pred,max_pred)
#loop para percorer o vetor com todos casos testados efetivamente
for r in range(0,len(utilidade),1):
    if (debug == 1):
        pyplot.plot(utilidade[r], label='APACHE II')
        pyplot.axis([0, max_pred, 0, max(result['apache t-1'])+1])
        pyplot.legend()
        pyplot.show()
        print(utilidade[r],'\n')
    positivo = 0
    #loop para percorer o vetor com as predições de cada hadm_id
    for p in range(0,max_pred-1,1):
        if (utilidade[r][p] == apache_alert):
            if (debug == 1):
                print('ALERTA VALOR')
            positivo = 1
        if ((utilidade[r][p+1] - utilidade[r][p]) > apache_delta):
            if (debug == 1):
                print('ALERTA DELTA')
            positivo = 1
    if (positivo == 1):
        alert_true = alert_true+1
        
    if (debug == 1):
        print('-------------')

print('Alert_true:', alert_true, 'de', len(utilidade))
indice_alert = round(float(alert_true)*100/(len(utilidade)),2)
print(indice_alert, '%')

