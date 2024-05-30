import os
import sys
import pandas as pd
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from preproc import DataLoader
from preproc import DataProcessor
from preproc import DataProcessor2
from preproc import DataProcessor3
from preproc import UnivariadoAnalyzer
from preproc import FeatureSelector
from preproc import DummyCreator
from preproc import ModelTrainer
from preproc import ComparadorColumnas
def main():
    excel_file_path = "../in/cacao.xlsx"
    df = pd.read_excel("../in/cacao.xlsx")
    data_loader = DataLoader(excel_file_path)
    if data_loader.load_excel():
        print(data_loader.data.head())
    else:
        print("No se pudieron cargar los datos.")
    data_processor = DataProcessor()
    data_processor2= DataProcessor2()
    data_processor3= DataProcessor3()
    data_processor4 = UnivariadoAnalyzer()
    data_processor5 = FeatureSelector()
    data_processor6 = DummyCreator()
    data_processor7 = ModelTrainer()
    data_processor8 = ComparadorColumnas()
    df = data_processor.convertir_tipos(df)
    print(df.head())
    df = data_processor2.categorizar_trimestre(df)
    df = data_processor2.categorizar_cosecha(df)
    df = data_processor2.mapear_categoria(df)
    df = data_processor2.llenar_na_continente_destino(df)
    df = data_processor2.escalar_variables(df)
    df = data_processor2.renombrar_columnas(df)
    df = data_processor2.filtrar_paises(df)
    df = data_processor3.procesar_datos(df)
    df = data_processor4.analisis_univariado(df)
    categorical_feat = list(df.select_dtypes(include=['category','object']).columns)
    df[categorical_feat] = df[categorical_feat].applymap(str)
    X = df.drop(['venta_fiable'], axis=1)
    y = df['venta_fiable'].round().astype(int)
    X_train, X_test, y_train, y_test, p_data = data_processor5.feature_selection(X, y)
    X_train = data_processor6.create_dummy_variables(X_train, list(p_data['Feature']))
    X_test = data_processor6.create_dummy_variables(X_test, list(p_data['Feature']))
    X_test = X_test.reindex(labels=X_train.columns, axis=1, fill_value=0)
    y_test = y_test.reindex(axis=1, fill_value=0)
    print(X_train)
    print(X_train.columns)
    print(X_test)
    in_put=pd.read_excel("../in/input.xlsx")
    in_put = data_processor.convertir_tipos(in_put)
    in_put = data_processor2.categorizar_trimestre(in_put)
    in_put = data_processor2.categorizar_cosecha(in_put)
    in_put = data_processor2.mapear_categoria(in_put)
    in_put = data_processor2.llenar_na_continente_destino(in_put)
    in_put = data_processor2.escalar_variables(in_put)
    in_put = data_processor2.renombrar_columnas(in_put)
    in_put = data_processor3.procesar_datos(in_put)
    in_put = in_put[['Peso_Kilos_brutos', 'Numero_articulos', 'Valor_FOB_USD', 'Valor_flete','Año', 'Valor_agregado_nacional', 'Valor_seguro', 'Cantidades', 'Valor_otros', 'Precio_unitario_FOB_COP_Peso_Neto', 'Dia', 'Modalidad_exportacion', 'Nacionalidad_medio_transporte', 'Agente_aduanero','Razon_social_exportador', 'Categoria', 'Certificado_origen','Lugar_salida', 'Pais_destino', 'Aduana', 'Departmanento_procedencia', 'Departamento_origen', 'Descripcion_partida_arancelaria', 'Forma_pago','Via_transporte','Sistemas_especiales','Continente_destino','Razon_social_importador']]
    in_put = data_processor6.create_dummy_variables(in_put, list(p_data['Feature']))
    in_put=data_processor8.comparar_y_agregar_columnas(X_train,in_put)
    print(in_put.columns)
    print(in_put)
    data_processor7.train_model(in_put,X_train, y_train, X_test, y_test)
    
    
if __name__ == "__main__":
    main()