import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway
from sklearn.feature_selection import f_classif
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix,recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, f1_score, confusion_matrix,recall_score
import time

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_excel(self):
        try:
            self.data = pd.read_excel(self.file_path)
            return True
        except Exception as e:
            print("Error al cargar el archivo:", e)
            return False
        
class DataProcessor:
   

    def convertir_tipos(self, df):
        df['Peso en kilos netos'] = df['Peso en kilos netos'].astype(float)
        df['Peso en kilos brutos'] = df['Peso en kilos brutos'].astype(float)
        df['Número de artículos'] = df['Número de artículos'].astype(float)
        df['Valor FOB (USD)'] = df['Valor FOB (USD)'].astype(float)
        df['Valor FOB (COP)'] = df['Valor FOB (COP)'].astype(float)
        df['Valor Agregado Nacional (VAN)'] = df['Valor Agregado Nacional (VAN)'].astype(float)
        df['Valor Flete'] = df['Valor Flete'].astype(float)
        df['Valor seguro'] = df['Valor seguro'].astype(float)
        df['Precio Unitario FOB (COP) Peso Neto'] = df['Precio Unitario FOB (COP) Peso Neto'].astype(float)
        df['Precio Unitario FOB (COP) Peso Bruto'] = df['Precio Unitario FOB (COP) Peso Bruto'].astype(float)
        df['Precio Unitario FOB (USD) Peso Neto'] = df['Precio Unitario FOB (USD) Peso Neto'].astype(float)
        df['Precio Unitario FOB (USD) Peso Bruto'] = df['Precio Unitario FOB (USD) Peso Bruto'].astype(float)
        df['Precio Unitario FOB (USD) Cantidad'] = df['Precio Unitario FOB (USD) Cantidad'].astype(float)
        df['Precio Unitario FOB (COP) Cantidad'] = df['Precio Unitario FOB (COP) Cantidad'].astype(float)
        df['Cantidad(es)'] = df['Cantidad(es)'].astype(float)
        
        return df       
    
class DataProcessor2:
    
    

    def asignar_trimestre(self, mes):
        if mes in [1,2,3]:
            return "Trimestre 1"
        elif mes in  [4,5,6]:
            return "Trimestre 2"
        elif mes in  [7,8,9]:
            return "Trimestre 3"
        else:
            return "Trimestre 4"

    def categorizar_trimestre(self, df):
        df['trimestre'] = df['Mes'].apply(self.asignar_trimestre)
        return df

    def asignar_cosecha(self, mes):
        if mes in [5,6,7,8]:
            return "cosecha principal"
        elif mes in  [11,12,1,2]:
            return "cosecha intermedia"
        else:
            return "No hay cosecha"

    def categorizar_cosecha(self, df):
        df['Cosecha'] = df['Mes'].apply(self.asignar_cosecha)
        return df

    def mapear_categoria(self, df):
        df['Categoria'] = df['Categoria'].str.strip()

        mapeo_cp ={
    'Cacao crudo en grano  entero para siembra':'Cacao crudo',
    'Cacao en polvo':'Cacao en polvo',
    'Cacao en polvo con adicion de azucar u otro edulcorante':'Cacao en polvo',
    'Cacao en polvo sin adicion de azucar ni otro edulcorante':'Cacao en polvo',
    'Cacao tostado en grano  entero o partido':'Cacao tostado',
    'Cascara  peliculas y demas residuos de cacao':'Cascara de cacao',
    'Las demas preparaciones alimenticias que contengan cacao  sin adicion de azucar  ni otros edulcorantes  en bloques o barras con peso superior a 2 kg  o en forma liquida  pastosa  en polvo  granulos o':'otras preparaciones',
    'Las demas preparciones alimenticias que contengan cacao  en bloques o barras con peso superior a 2 kg  o en forma liquida  pastosa  en polvo  granulos o en formas similares':'otras preparaciones',
    'Las demas preparciones alimenticias que contengan cacao  en bloques o barras con peso superior a 2 kg  o en forma liquida  pastosa  en polvo  granulos o en formas similares  en recipientes o envases i':'otras preparaciones',
    'Los demas cacaos crudos en grano  entero o partido':'Cacao crudo',
    'Los demas chocolates y demas preparaciones alimenticas que contengan cacao  en bloques  tabletas o barras  sin rellenar':'Chocolates',
    'Los demas chocolates y demas preparaciones alimenticias que contengan cacao':'Chocolates',
    'Los demas chocolates y demas preparaciones alimenticias que contengan cacao  en bloques  tabletas o barras  rellenos':'Chocolates',
    'Los demas chocolates y demas preparaciones alimenticias que contengan cacao  en bloques  tabletas o barras  sin rellenar  sin adicion de azucar  ni otros edulcorantes':'Chocolates',
    'Los demas chocolates y demas preparaciones alimenticias que contengan cacao  sin adicion de azucar  ni otros edulcorantes':'Chocolates',
    'Manteca de cacao  con un índice de acidez expresado en ácido oleico':'Manteca de cacao',
    'Manteca de cacao  con un Indice de acidez expresado en acido oleico inferior o igual a 1 porciento':'Manteca de cacao',
    'Manteca de cacao  con un indice de acidez expresado en acido oleico superior a 1 porciento pero inferior o igual a 165 porciento':'Manteca de cacao',
    'NA':'otras preparaciones',
    'Pasta de cacao desgrasada total o parcialmente':'pasta de cacao',
    'Pasta de cacao sin desgrasar':'pasta de cacao',
        }
        df['Categoria'] = df['Categoria'].replace(mapeo_cp)
        print(df['Categoria'].value_counts())
        return df

    def llenar_na_continente_destino(self, df):
        df['Continente Destino'].fillna('no determinado', inplace=True)
        return df

    def escalar_variables(self, df):
        scaler = MinMaxScaler()
        variables_a_escalar = ['Cantidad(es)', 'Peso en kilos brutos', 'Número de artículos',
                               'Valor FOB (COP)', 'Valor Agregado Nacional (VAN)', 'Valor Flete',
                               'Valor seguro', 'Valor otros', 'Precio Unitario FOB (COP) Peso Neto',
                               'Precio Unitario FOB (COP) Peso Bruto', 'Precio Unitario FOB (USD) Peso Neto',
                               'Precio Unitario FOB (USD) Peso Bruto', 'Precio Unitario FOB (USD) Cantidad',
                               'Precio Unitario FOB (COP) Cantidad']
        df[variables_a_escalar] = scaler.fit_transform(df[variables_a_escalar])
        return df

    def renombrar_columnas(self, df):
        # Renombrar columnas
        df = df.rename(columns={'Tipo de declaración': 'Tipo_de_declaracion',
        'Agente aduanero(s)': 'Agente_aduanero',
        'Razón social actual Exportador': 'Razon_social_exportador',
        'Razón social del importador': 'Razon_social_importador',
        'Código Partida': 'Codigo_partida',
        'Descripción de la partida arancelaria': 'Descripcion_partida_arancelaria',
        'Cantidad(es)': 'Cantidades',
        'Peso en kilos netos': 'Peso_kilos_netos',
        'Peso en kilos brutos': 'Peso_Kilos_brutos',
        'Número de artículos': 'Numero_articulos',
        'País de Destino': 'Pais_destino',
        'Departamento Origen': 'Departamento_origen',
        'Departamento De Procedencia': 'Departmanento_procedencia',
        'Lugar de salida': 'Lugar_salida',
        'Vía de transporte': 'Via_transporte',
        'Nacionalidad del medio de transporte': 'Nacionalidad_medio_transporte',
        'Regimen Exportacion': 'Regimen_exportacion',
        'Modalidad de exportación': 'Modalidad_exportacion',
        'Certificado de Origen': 'Certificado_origen',
        'Sistemas Especiales': 'Sistemas_especiales',
        'Forma de pago': 'Forma_pago',
        'Valor FOB (USD)': 'Valor_FOB_USD',
        'Valor FOB (COP)': 'Valor_FOB_COP',
        'Valor Agregado Nacional (VAN)': 'Valor_agregado_nacional',
        'Valor Flete': 'Valor_flete',
        'Valor seguro': 'Valor_seguro',
        'Valor otros': 'Valor_otros',
        'Precio Unitario FOB (COP) Peso Neto': 'Precio_unitario_FOB_COP_Peso_Neto',
        'Precio Unitario FOB (COP) Peso Bruto': 'Precio_unitario_FOB_COP_Peso_Bruto',
        'Precio Unitario FOB (USD) Peso Neto': 'Precio_unitario_FOB_USD_peso_Neto',
        'Precio Unitario FOB (USD) Peso Bruto': 'Precio_unitario_FOB_USD_Peso_Bruto',
        'Precio Unitario FOB (USD) Cantidad': 'Precio_Unitario_FOB_USD_Cantidad',
        'Precio Unitario FOB (COP) Cantidad': 'Precio_unitario_FOB_COP_Cantidad',
        'Continente Destino': 'Continente_destino'
                                
                                })
        return df

    def filtrar_paises(self, df):
        temp_co = df['Pais_destino'].value_counts()
        pais_count = df['Pais_destino'].apply(lambda x: temp_co[x])
        for i in range(0, len(df)):
            if pais_count[i] < pais_count.quantile(0.05):
                df['Pais_destino'][i] = 'OTROS PAISES'
        df = df[~df['Pais_destino'].isin(['COLOMBIA', 'ZONA FRANCA  DE  BOGOTA', 'OTROS PAISES'])]
        return df 
    
    
class DataProcessor3:
   

    def calcular_calificacion(self, row):
        categoria = str(row['Categoria'])  
        peso_kilos_netos = row['Peso_kilos_netos']
        var_fob_dolar = row['Valor_FOB_USD']

        rangos_calificaciones = {
        'Chocolates': {
            'peso': {
                (0, 579): 0,
                (580 , float('inf')): 1
            },
            'fob': {
                (0, 2499): 0,
                (2500 , float('inf')): 1
            }
        },
        'Cacao en polvo': {
             'fob': {
                (0, 3999): 0,
                (4000 , float('inf')): 1
            },
            'peso': {
                (0, 1799): 0,
                (1800 , float('inf')): 1
            }
        },
        'Cacao crudo': {
             'peso': {
                (0, 24999): 0,
                (25000 , float('inf')): 1
            },
            'fob': {
                (0, 62999): 0,
                (63000 , float('inf')): 1
            }
        },
        'Manteca de cacao': {
             'peso': {
                (0, 7999): 0,
                (8000, float('inf')): 1
            },
            'fob': {
                (0, 44999): 0,
                (45000 , float('inf')): 1
            }
        },
        'pasta de cacao': {
            'peso': {
                (0, 7999): 0,
                (8000 , float('inf')): 1
            },
            'fob': {
                (0, 31999): 0,
                (32000 , float('inf')): 1
            }
        },
        'Cacao tostado': {
            'peso': {
                (0, 599): 0,
                (600 , float('inf')): 1
            },
            'fob': {
                (0, 3099): 0,
                (3100  , float('inf')): 1
            }
        },
        'otras preparaciones': {
             'peso': {
                (0, 949): 0,
                (950 , float('inf')): 1
            },
            'fob': {
                (0, 3899 ): 0,
                (3900  , float('inf')): 1
            }
        },
        'Cascara de cacao': {
             'peso': {
                (0, 12599): 0,
                (12600 , float('inf')): 1
            },
            'fob': {
                (0, 6599 ): 0,
                (6600 , float('inf')): 1
            }
        }
    }

        if categoria in rangos_calificaciones:
            calificacion_peso = None
            calificacion_fob = None
            
            for rango, calificacion in rangos_calificaciones[categoria]['peso'].items():
                if rango[0] <= peso_kilos_netos <= rango[1]:
                    calificacion_peso = calificacion
                    break

            for rango, calificacion in rangos_calificaciones[categoria]['fob'].items():
                if rango[0] <= var_fob_dolar <= rango[1]:
                    calificacion_fob = calificacion
                    break

            if calificacion_peso is None or calificacion_fob is None:
                return None
            
            calificacion_promedio = (calificacion_peso + calificacion_fob) / 2
            
            if calificacion_promedio >=0.5:
                return 1
            else:
                return 0
        else:
            return 0

    def procesar_datos(self, df):
        
         # Calculamos la columna 'venta_fiable' y la asignamos al DataFrame
        df['venta_fiable'] = df.apply(self.calcular_calificacion, axis=1)
        df['venta_fiable'] = df['venta_fiable'].fillna(0)
        print(df['venta_fiable'].value_counts())
        # Escalamos las variables 'Peso_kilos_netos' y 'Valor_FOB_USD'
        scaler = MinMaxScaler()
        df[['Peso_kilos_netos']] = scaler.fit_transform(df[['Peso_kilos_netos']])
        df[['Valor_FOB_USD']] = scaler.fit_transform(df[['Valor_FOB_USD']])

       
        return df   
    
class UnivariadoAnalyzer:
    
    
    def analisis_univariado(self, df):
            categorical_feat = list(df.select_dtypes(include=['category','object']).columns)
            df[categorical_feat] = df[categorical_feat].applymap(str)
            df[categorical_feat].nunique().reset_index().sort_values(by=0, ascending=False)
            temp_razonimp = df['Razon_social_importador'].value_counts()
            temp_razonexp = df['Razon_social_exportador'].value_counts()
            temp_agentead = df['Agente_aduanero'].value_counts()
            temp_nacmedtramp = df['Nacionalidad_medio_transporte'].value_counts()
            temp_desparara = df['Descripcion_partida_arancelaria'].value_counts()
            temp_depor=df['Departamento_origen'].value_counts()
            temp_deppro=df['Departmanento_procedencia'].value_counts()
            temp_aduana=df['Aduana'].value_counts()
            temp_lugsal=df['Lugar_salida'].value_counts()
            temp_modexp=df['Modalidad_exportacion'].value_counts()
            razonimp_count = df['Razon_social_importador'].apply(lambda x: temp_razonimp[x])
            razonexp_count = df['Razon_social_exportador'].apply(lambda x: temp_razonexp[x])
            agentead_count = df['Agente_aduanero'].apply(lambda x: temp_agentead[x])
            nacmedtramp_count = df['Nacionalidad_medio_transporte'].apply(lambda x: temp_nacmedtramp[x])
            desparara_count =df['Descripcion_partida_arancelaria'].apply(lambda x: temp_desparara[x])
            depor_count = df['Departamento_origen'].apply(lambda x: temp_depor[x])
            deppro_count = df['Departmanento_procedencia'].apply(lambda x: temp_deppro[x])
            aduana_count = df['Aduana'].apply(lambda x: temp_aduana[x])
            lugsal_count = df['Lugar_salida'].apply(lambda x: temp_lugsal[x])
            modexp_count = df['Modalidad_exportacion'].apply(lambda x: temp_modexp[x])
            quantile_20 = razonimp_count.quantile(0.2)
            df.loc[razonimp_count < quantile_20, 'Razon_social_importador'] = 'OTRO IMPORTADOR'
            quantile_20 = razonexp_count.quantile(0.2)
            df.loc[razonexp_count < quantile_20, 'Razon_social_exportador'] = 'OTRO EXPORTADOR'
            quantile_20 = agentead_count.quantile(0.2)
            df.loc[agentead_count < quantile_20, 'Agente_aduanero'] = 'OTRO AGENTE'
            quantile_10 = nacmedtramp_count.quantile(0.1)
            df.loc[nacmedtramp_count < quantile_10, 'Nacionalidad_medio_transporte'] = 'OTRO NACIONALIDAD'
            quantile_20 = desparara_count.quantile(0.2)
            df.loc[desparara_count < quantile_20, 'Descripcion_partida_arancelaria'] = 'OTRA DESCRIPCION'
            quantile_10 = depor_count.quantile(0.1)
            df.loc[depor_count < quantile_10, 'Departamento_origen'] = 'OTRO DEPARTAMENTO'
            quantile_10 = deppro_count.quantile(0.1)
            df.loc[deppro_count < quantile_10, 'Departmanento_procedencia'] = 'OTRO DEPARTAMENTO'
            quantile_10 = aduana_count.quantile(0.1)
            df.loc[aduana_count < quantile_10, 'Aduana'] = 'OTRA ADUANA'
            quantile_10 = lugsal_count.quantile(0.1)
            df.loc[lugsal_count < quantile_10, 'Lugar_salida'] = 'OTRO LUGAR'
            quantile_20 = modexp_count.quantile(0.2)
            df.loc[modexp_count < quantile_20, 'Modalidad_exportacion'] = 'OTRA MODALIDAD'
            df[categorical_feat].nunique().reset_index().sort_values(by=0, ascending=False)

            
            return df 
 
class FeatureSelector:
    

    def feature_selection(self, X, y):
        # Dividir X y usando Train test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1727, stratify=y)
        X_train_num = X_train.select_dtypes(include='number').copy()

        # Definir un diccionario vacío para almacenar los resultados de la prueba chi-cuadrado
        chi2_check = {}

        # Recorrer cada columna en el conjunto de entrenamiento para calcular la estadística chi con la variable objetivo
        X_train_cat = X_train.select_dtypes(include=['object', 'category']).copy()
        for column in X_train_cat:
            chi, p, dof, ex = chi2_contingency(pd.crosstab(y_train, X_train_cat[column]))
            chi2_check.setdefault('Feature', []).append(column)
            chi2_check.setdefault('p-value', []).append(round(p, 10))

        # Convertir diccionario a DataFrame
        chi2_result = pd.DataFrame(data=chi2_check)

        data_merge = chi2_result.merge(X_train_cat.describe().T.reset_index(),
                                        left_on='Feature',
                                        right_on='index').sort_values(by=['p-value', 'unique'])

        p_data = data_merge[(data_merge['p-value'] < 0.05)].sort_values(by='p-value')

        # ANOVA
        f_statistics, p_values = f_classif(X_train_num.fillna(X_train_num.median()), y_train)
        anova_f_table = pd.DataFrame(data={'Feature': X_train_num.columns.values,
                                           'F-Score': f_statistics,
                                           'p-value': p_values.round(decimals=10)})
        anova_merge = anova_f_table.merge(X_train_num.describe().T.reset_index(),
                                          left_on='Feature',
                                          right_on='index').sort_values(['F-Score', 'count'], ascending=False).head(
            50)

        p_anova = anova_merge[(anova_merge['p-value'] < 0.05)].sort_values(by='p-value')

        # Correlación
        corr_matrix = X_train[list(p_anova['Feature'])].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
        num_consider = [x for x in list(p_anova['Feature']) if x not in to_drop]
        selected_cols = num_consider + list(p_data['Feature'])
        X_train = X_train[selected_cols]
        X_test = X_test[selected_cols]
        columnas_a_eliminar = ['Agente_aduanero', 'Razon_social_exportador', 'Razon_social_importador',
                               'Nacionalidad_medio_transporte']
        X_train = X_train.drop(columns=columnas_a_eliminar)
        X_test = X_test.drop(columns=columnas_a_eliminar)
        return X_train, X_test, y_train, y_test, p_data 
    
    
class DummyCreator:
    
    def create_dummy_variables(self, df, cols):
        # Verificar qué columnas están presentes en el DataFrame
        present_cols = [col for col in cols if col in df.columns]
        
        if not present_cols:
            print("Ninguna de las columnas especificadas está presente en el DataFrame.")
            return df
        
        # Obtener variables dummy solo para las columnas presentes
        df_dummies = pd.get_dummies(df[present_cols], prefix_sep=':')
        df_dummies = df_dummies.astype(int) # convertir a valores enteros
        
        # Eliminar las columnas originales del DataFrame
        df = df.drop(columns=present_cols)
        
        # Concatenar las nuevas columnas dummy con el DataFrame original
        df = pd.concat([df, df_dummies], axis=1)
        
        return df
    


    
class ModelTrainer:
    
    def train_model(self,df ,X_train, y_train, X_test, y_test, k=68):
        start_time = time.time()
       
        if k == 68:
            # Crear y entrenar el modelo
            model = MLPClassifier(
                hidden_layer_sizes=(106, k, 1),
                activation='identity',  # Sigmoide
                solver='adam',
                random_state=1
            )
            model.fit(X_train, y_train)
            # Realizar predicciones en el conjunto de prueba
            Y_pred = model.predict(X_test)
            
            # Calcular y mostrar la precisión del modelo
            accuracy = accuracy_score(y_test, Y_pred)
            mse = mean_squared_error(y_test, Y_pred)
            cm = confusion_matrix(y_test, Y_pred)
            f1 = f1_score(y_test, Y_pred)
            recall = recall_score(y_test, Y_pred)
            
            if accuracy > 0.8:
                print(f'Para k={k}, La precisión del modelo: {accuracy}')
                print(f'Para k={k}, MSE del modelo: {mse}')
                print(f'Para k={k}, F1-score del modelo: {f1}')
                print(f'Para k={k}, Recall del modelo: {recall}')
                print(f'Para k={k}, Matriz de confusión:')
                print(cm)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f'Tiempo de ejecución: {execution_time} segundos')
        Y_pred2 = model.predict(df)
        if Y_pred2 == 1:
            print("la exportación es fiable")
            print(Y_pred2)
        else:
            print("la exportación no es fiable")
            print(Y_pred2)
                
        
        

 
class Dummies:
    def __init__(self, df):
        self.df = df
    
    def apply_get_dummies(self, columns=None):
        
        if columns is None:
            # Si no se especifican columnas, aplicar get_dummies a todas las columnas categóricas
            categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        else:
            # Si se especifican columnas, solo aplicar get_dummies a esas columnas
            categorical_cols = columns
        
        # Aplicar get_dummies
        df = pd.get_dummies(self.df, columns=categorical_cols)
        
        df = df.astype(int)
        print (df)
        print(df.columns)  
        return df
    
class ComparadorColumnas:
   
    
    def comparar_y_agregar_columnas(self,df1,df2):
        # Obtener columnas faltantes
        columnas_faltantes = df1.columns.difference(df2.columns)
        
        # Agregar columnas faltantes al DataFrame destino
        for columna in columnas_faltantes:
            df2[columna] = 0 # o cualquier otro valor por defecto que desees
            
        df2 = df2[df1.columns]
        return df2    
    
