import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from dotenv import load_dotenv
from openai import OpenAI
from io import BytesIO

# Configuración de la página
st.set_page_config(
    page_title="BCRP Exchange Rate Analysis",
    page_icon="📈",
    layout="wide"
)

# Cargar variables de entorno
load_dotenv()

# Agregar un diccionario para traducir los meses en español
MESES_ESP_A_ING = {
    'Ene': 'Jan',
    'Feb': 'Feb',
    'Mar': 'Mar',
    'Abr': 'Apr',
    'May': 'May',
    'Jun': 'Jun',
    'Jul': 'Jul',
    'Ago': 'Aug',
    'Set': 'Sep',
    'Sep': 'Sep',
    'Oct': 'Oct',
    'Nov': 'Nov',
    'Dic': 'Dec'
}

def convertir_fecha_esp_a_ing(fecha_esp):
    """Convierte una fecha en formato español a formato inglés"""
    try:
        dia, mes, anio = fecha_esp.split('.')
        mes_ing = MESES_ESP_A_ING[mes]
        return f"{dia}.{mes_ing}.{anio}"
    except (KeyError, ValueError) as e:
        print(f"Error procesando fecha: {fecha_esp}, Error: {str(e)}")
        return fecha_esp  # Devolver la fecha original si hay error

def get_dataframe(series_code, series_name, start_period, end_period):
    url = f"https://estadisticas.bcrp.gob.pe/estadisticas/series/api/{series_code}/json/{start_period}/{end_period}"
    
    try:
        response = requests.get(url)
        if response.status_code == 403:
            return None
        response.raise_for_status()
        
        data = response.json()
        records = data.get('periods', [])
        
        if not records:
            return None
            
        # Filtrar y procesar los datos
        valid_records = []
        for record in records:
            try:
                if record['values'][0] != 'n.d.' and record['values'][0] is not None:
                    valid_records.append({
                        'Date': record['name'],
                        series_name: float(record['values'][0])
                    })
            except (ValueError, TypeError):
                continue
        
        if not valid_records:
            return None
            
        # Crear DataFrame con los registros válidos
        df = pd.DataFrame(valid_records)
        return df
        
    except Exception as e:
        st.error(f"Error al obtener datos: {str(e)}")
        return None

class EconomicChatbot:
    def __init__(self, exchange_rates_df):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            st.error("OpenAI API key not found. Please add OPENAI_API_KEY to your secrets.")
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key)
        self.conversation_history = []
        self.exchange_rates = exchange_rates_df
    
    def get_context_prompt(self):
        # Crear un resumen más detallado de los datos
        df = self.exchange_rates
        
        context = f"""
        Eres un asistente especializado en análisis del tipo de cambio PEN/USD del Banco Central de Reserva del Perú.
        
        Datos disponibles del {df['Date'].min().strftime('%Y-%m-%d')} al {df['Date'].max().strftime('%Y-%m-%d')}
        
        Información actual:
        - Tipo de cambio más reciente: {df.iloc[-1]['Exchange Rate (PEN to USD)']:.3f} PEN/USD
        - Tipo de cambio más alto: {df['Exchange Rate (PEN to USD)'].max():.3f} PEN/USD
        - Tipo de cambio más bajo: {df['Exchange Rate (PEN to USD)'].min():.3f} PEN/USD
        - Promedio del período: {df['Exchange Rate (PEN to USD)'].mean():.3f} PEN/USD
        
        Serie completa de datos (fecha: tipo de cambio):
        {df.set_index('Date')['Exchange Rate (PEN to USD)'].to_dict()}
        
        Por favor, responde preguntas sobre estos datos de manera precisa y directa.
        Si te preguntan por datos fuera de este rango, indica que no tienes esa información disponible.
        """
        return context
    
    def get_streaming_response(self, prompt):
        try:
            if self.client is None:
                st.error("OpenAI client not initialized. Please configure your API key.")
                return None
                
            full_prompt = self.get_context_prompt() + "\n\nHistorial de conversación:\n"
            
            for msg in self.conversation_history[-3:]:
                if msg["role"] == "user":
                    full_prompt += f"\nHuman: {msg['content']}"
                else:
                    full_prompt += f"\nAssistant: {msg['content']}"
            
            full_prompt += f"\nHuman: {prompt}\nAssistant:"
            
            response = self.client.chat.completions.create(
                model="o4-mini-2025-04-16",
                messages=[
                    {"role": "user", "content": full_prompt.strip()}
                ],
                stream=True
            )
            
            return response
            
        except Exception as e:
            st.error(f"Error en el chatbot: {str(e)}")
            return None
    
    def update_history(self, prompt, response):
        self.conversation_history.extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ])

class ReportGenerator:
    def __init__(self, exchange_rates_df):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            st.error("OpenAI API key not found. Please add OPENAI_API_KEY to your secrets.")
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key)
        self.exchange_rates = exchange_rates_df
    
    def get_context_prompt(self):
        df = self.exchange_rates
        
        context = f"""
        Genera un informe detallado sobre el tipo de cambio PEN/USD basado en los siguientes datos:
        
        Período analizado: {df['Date'].min().strftime('%Y-%m-%d')} al {df['Date'].max().strftime('%Y-%m-%d')}
        
        Métricas clave:
        - Tipo de cambio actual: {df.iloc[-1]['Exchange Rate (PEN to USD)']:.3f} PEN/USD
        - Tipo de cambio máximo: {df['Exchange Rate (PEN to USD)'].max():.3f} PEN/USD
        - Tipo de cambio mínimo: {df['Exchange Rate (PEN to USD)'].min():.3f} PEN/USD
        - Promedio del período: {df['Exchange Rate (PEN to USD)'].mean():.3f} PEN/USD
        - Desviación estándar: {df['Exchange Rate (PEN to USD)'].std():.3f}
        
        El informe debe incluir las siguientes secciones:
        1. Executive Summary (Always mention the date range of the data)
        2. Context Overview
        3. Dataset Summary
        4. Trend Analysis
        5. Key Insights
        6. Final Thoughts
        
        Por favor, proporciona un análisis detallado y profesional basado en estos datos.
        """
        return context
    
    def generate_report(self):
        try:
            if self.client is None:
                st.error("OpenAI client not initialized. Please configure your API key.")
                return None
                
            response = self.client.chat.completions.create(
                model="o4-mini-2025-04-16",
                messages=[
                    {"role": "user", "content": "Actúa como un analista financiero experto del BCRP especializado en análisis del tipo de cambio. " + self.get_context_prompt()}
                ],
                stream=True
            )
            return response
        except Exception as e:
            st.error(f"Error generando el reporte: {str(e)}")
            return None

def main():
    st.title("📊 Análisis del Tipo de Cambio BCRP")
    
    # Obtener datos iniciales para establecer el rango de fechas disponible
    if 'exchange_rate_df' not in st.session_state:
        initial_df = get_dataframe("PD04640PD", "Exchange Rate (PEN to USD)", "2023-01-01", datetime.now().strftime("%Y-%m-%d"))
        if initial_df is not None:
            try:
                initial_df['Date'] = initial_df['Date'].apply(convertir_fecha_esp_a_ing)
                initial_df['Date'] = pd.to_datetime(initial_df['Date'], format='%d.%b.%y')
                st.session_state.exchange_rate_df = initial_df
            except Exception as e:
                st.error(f"Error al procesar las fechas iniciales: {str(e)}")
                return
    
    # Obtener el rango de fechas disponible
    if 'exchange_rate_df' in st.session_state:
        min_date = st.session_state.exchange_rate_df['Date'].min().date()
        max_date = st.session_state.exchange_rate_df['Date'].max().date()
    else:
        min_date = datetime(2023, 1, 1).date()
        max_date = datetime.now().date()
    
    # Sidebar con información del proyecto
    st.sidebar.title("ℹ️ Acerca del Proyecto")
    
    st.sidebar.markdown("""
    ### BCRP Exchange Rate Analytics
    
    Esta aplicación proporciona análisis avanzado del tipo de cambio PEN/USD utilizando datos oficiales del Banco Central de Reserva del Perú (BCRP).
    
    #### Características Principales:
    - 📊 Visualización interactiva de datos
    - 📈 Análisis de tendencias
    - 🤖 Asistente AI para consultas
    - 📑 Generación de reportes automáticos
    
    #### Datos Disponibles
    Los datos se actualizan diariamente desde la API oficial del BCRP.
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Desarrollado por")
    st.sidebar.markdown("""
    Patricio Rios Cánepa
                        
    Linkedin Profile: https://www.linkedin.com/in/patriciorioscanepa/
                        
    Github Profile: https://github.com/PatricRc
                        
    Personal Website: https://data-pat-ai.netlify.app/
    """)
    
    # Botón de actualización en la barra lateral
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Actualizar Datos", key="update_button", help="Actualiza los datos al último disponible"):
        if 'exchange_rate_df' in st.session_state:
            del st.session_state.exchange_rate_df
        if 'chatbot' in st.session_state:
            del st.session_state.chatbot
    
    # Usar las fechas del gráfico como fechas principales
    start_date = min_date
    end_date = max_date
    
    # Validación de fechas
    if start_date > end_date:
        st.error("❌ La fecha de inicio debe ser anterior a la fecha final")
        return
    
    # Convertir fechas al formato requerido por la API
    start_period = start_date.strftime("%Y-%m-%d")
    end_period = end_date.strftime("%Y-%m-%d")
    
    # Obtener datos solo si no están en session_state o si se presionó el botón de actualización
    if 'exchange_rate_df' not in st.session_state:
        exchange_rate_df = get_dataframe("PD04640PD", "Exchange Rate (PEN to USD)", start_period, end_period)
        
        if exchange_rate_df is not None:
            try:
                # Convertir las fechas de español a inglés antes de la conversión
                exchange_rate_df['Date'] = exchange_rate_df['Date'].apply(convertir_fecha_esp_a_ing)
                exchange_rate_df['Date'] = pd.to_datetime(exchange_rate_df['Date'], format='%d.%b.%y')
                
                # Guardar el DataFrame en session_state
                st.session_state.exchange_rate_df = exchange_rate_df
                
            except Exception as e:
                st.error(f"Error al procesar las fechas: {str(e)}")
                st.write("Ejemplos de fechas:", exchange_rate_df['Date'].head())
                return
    else:
        exchange_rate_df = st.session_state.exchange_rate_df

    if exchange_rate_df is not None:
        # Mostrar datos
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📈 Gráfico del Tipo de Cambio Diario")
            st.markdown("*Visualización interactiva del tipo de cambio PEN/USD. Utiliza los filtros de fecha para analizar períodos específicos.*")
            
            # Agregar filtros adicionales para el gráfico
            chart_container = st.container()
            with chart_container:
                # Date pickers para el filtro del gráfico
                filter_col1, filter_col2 = st.columns(2)
                with filter_col1:
                    chart_start_date = st.date_input(
                        "Filtrar gráfico desde",
                        start_date,
                        min_value=min(exchange_rate_df['Date']).date(),
                        max_value=max(exchange_rate_df['Date']).date(),
                        key="chart_start_date",
                        help="Filtrar la visualización de datos desde esta fecha"
                    )
                with filter_col2:
                    chart_end_date = st.date_input(
                        "Filtrar gráfico hasta",
                        end_date,
                        min_value=min(exchange_rate_df['Date']).date(),
                        max_value=max(exchange_rate_df['Date']).date(),
                        key="chart_end_date",
                        help="Filtrar la visualización de datos hasta esta fecha"
                    )
                
                # Mostrar rango de fechas seleccionado
                st.caption(f"📅 Mostrando datos del {chart_start_date.strftime('%d/%m/%Y')} al {chart_end_date.strftime('%d/%m/%Y')}")
                
                # Filtrar datos según las fechas seleccionadas
                mask = (exchange_rate_df['Date'].dt.date >= chart_start_date) & \
                       (exchange_rate_df['Date'].dt.date <= chart_end_date)
                filtered_df = exchange_rate_df.loc[mask].copy()
            
            # Usar directamente los datos filtrados según las fechas principales
            if len(filtered_df) > 0:
                # Crear gráfico con los datos
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.lineplot(data=filtered_df, x='Date', y='Exchange Rate (PEN to USD)', ax=ax)
                plt.xticks(rotation=45)
                plt.title(f"Evolución del Tipo de Cambio Diario PEN/USD\n({chart_start_date.strftime('%d/%m/%Y')} - {chart_end_date.strftime('%d/%m/%Y')})")
                plt.grid(True, linestyle='--', alpha=0.7)
                st.pyplot(fig)
                plt.close()
            else:
                st.warning("No hay datos disponibles para el rango de fechas seleccionado")
        
        with col2:
            st.subheader("📋 Datos Recientes")
            st.markdown(f"*Tabla de los últimos registros del tipo de cambio del {chart_start_date.strftime('%d/%m/%Y')} al {chart_end_date.strftime('%d/%m/%Y')}. Ajusta el número de filas para ver más o menos datos.*")
            
            # Selector de número de filas a mostrar
            num_rows = st.number_input(
                "Número de registros a mostrar",
                min_value=1,
                max_value=len(filtered_df),
                value=min(12, len(filtered_df)),
                key="num_rows"
            )
            
            # Mostrar tabla filtrada
            if len(filtered_df) > 0:
                display_df = filtered_df.copy()
                display_df['Date'] = display_df['Date'].dt.strftime('%d.%b.%Y')
                st.dataframe(
                    display_df.tail(num_rows),
                    hide_index=True,
                    use_container_width=True
                )
                
                # Botón para descargar datos en Excel
                def convert_df_to_excel():
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        display_df.to_excel(writer, sheet_name='Tipo_de_Cambio', index=False)
                    return output.getvalue()
                
                excel_data = convert_df_to_excel()
                st.download_button(
                    label="📥 Descargar datos en Excel",
                    data=excel_data,
                    file_name=f'tipo_cambio_{chart_start_date.strftime("%Y%m%d")}_a_{chart_end_date.strftime("%Y%m%d")}.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
                
                # Mostrar estadísticas básicas
                st.subheader("📊 Estadísticas")
                st.markdown(f"*Resumen estadístico del período del {chart_start_date.strftime('%d/%m/%Y')} al {chart_end_date.strftime('%d/%m/%Y')}.*")
                stats_df = pd.DataFrame({
                    'Métrica': ['Promedio', 'Mínimo', 'Máximo', 'Desviación Estándar'],
                    'Valor': [
                        f"{filtered_df['Exchange Rate (PEN to USD)'].mean():.3f}",
                        f"{filtered_df['Exchange Rate (PEN to USD)'].min():.3f}",
                        f"{filtered_df['Exchange Rate (PEN to USD)'].max():.3f}",
                        f"{filtered_df['Exchange Rate (PEN to USD)'].std():.3f}"
                    ]
                })
                st.dataframe(stats_df, hide_index=True, use_container_width=True)
            else:
                st.warning("No hay datos disponibles para mostrar")
        
        # Sección del Chatbot
        st.markdown("---")
        st.subheader("💬 Consulta sobre el Tipo de Cambio")
        st.markdown("*Asistente AI especializado en análisis del tipo de cambio. Realiza preguntas sobre tendencias, comparaciones o análisis específicos.*")
        
        # Inicializar el chatbot con los datos filtrados
        if 'chatbot' not in st.session_state:
            st.session_state.chatbot = EconomicChatbot(exchange_rate_df)
        else:
            # Actualizar los datos del chatbot cuando cambia el filtro
            st.session_state.chatbot.exchange_rates = exchange_rate_df
        
        # Input del usuario
        user_input = st.text_input("Hazme una pregunta sobre el tipo de cambio:", key="user_input")
        
        if st.button("Enviar"):
            if user_input:
                response_container = st.empty()
                full_response = ""
                
                # Obtener respuesta streaming
                response = st.session_state.chatbot.get_streaming_response(user_input)
                
                if response:
                    for chunk in response:
                        if chunk.choices[0].delta.content is not None:
                            full_response += chunk.choices[0].delta.content
                            response_container.markdown(full_response)
                    
                    # Actualizar historial
                    st.session_state.chatbot.update_history(user_input, full_response)

        # Nueva sección para el Reporte AI
        st.markdown("---")
        st.subheader("📑 Reporte AI del Tipo de Cambio")
        
        # Agregar descripción con fechas seleccionadas
        st.markdown(f"*Genera un informe detallado para el período del {chart_start_date.strftime('%d/%m/%Y')} al {chart_end_date.strftime('%d/%m/%Y')}. Incluye resumen ejecutivo, análisis de tendencias e insights clave.*")
        
        if st.button("Generar Reporte"):
            # Usar filtered_df en lugar de exchange_rate_df
            report_generator = ReportGenerator(filtered_df)
            report_container = st.empty()
            full_report = ""
            
            with st.spinner('Generando reporte...'):
                response = report_generator.generate_report()
                
                if response:
                    for chunk in response:
                        if chunk.choices[0].delta.content is not None:
                            full_report += chunk.choices[0].delta.content
                            report_container.markdown(full_report)
                    
                    # Agregar botón para descargar el reporte
                    st.download_button(
                        label="📥 Descargar Reporte",
                        data=full_report,
                        file_name=f'reporte_tipo_cambio_{chart_start_date.strftime("%Y%m%d")}_a_{chart_end_date.strftime("%Y%m%d")}.txt',
                        mime='text/plain'
                    )
    else:
        st.error("No se pudieron obtener datos para el período seleccionado.")

if __name__ == "__main__":
    main()

