# BCRP Exchange Rate Analysis App

![BCRP Exchange Rate Analysis](https://img.shields.io/badge/BCRP-Exchange%20Rate%20Analysis-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat&logo=openai&logoColor=white)

A comprehensive Streamlit application for analyzing PEN/USD exchange rates using official data from the Central Reserve Bank of Peru (BCRP).

## 📊 Features

- **Interactive Data Visualization**: Explore exchange rate trends with customizable date ranges
- **Real-time Data**: Access the latest exchange rate data from BCRP's official API
- **AI-powered Analysis**: Ask questions about exchange rate trends and receive instant insights
- **Automated Reporting**: Generate comprehensive reports with key metrics and trends
- **Data Export**: Download exchange rate data in Excel format for further analysis

## 🚀 Demo

![BCRP Exchange Rate Analysis Demo](demo-screenshot.png)

## 🛠️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## 💻 Usage

1. Start the Streamlit app:
   ```bash
   streamlit run BRCP_API/PExchangeApp.py
   ```

2. Open your browser and navigate to the provided URL (typically http://localhost:8501)

3. Use the date filters to analyze specific time periods

4. Ask questions about exchange rate trends using the AI assistant

5. Generate and download reports for your selected date range

## 📌 Key Components

- **Interactive Charts**: Visualize daily exchange rate fluctuations
- **Recent Data Table**: View and filter the most recent exchange rate data
- **Statistical Summary**: Quick access to key metrics (min, max, average, standard deviation)
- **AI Chatbot**: Ask questions about exchange rate trends in natural language
- **Report Generator**: Create comprehensive analysis reports with a single click

## 🧠 AI Capabilities

The app includes two AI-powered features:
1. **Exchange Rate Chatbot**: Ask specific questions about trends, comparisons, or historical data
2. **Report Generator**: Create detailed reports with executive summary, trend analysis, and key insights

## 📄 Data Source

All exchange rate data is retrieved from the official Banco Central de Reserva del Perú (BCRP) API.

## 📝 License

[MIT License](LICENSE)

## 👨‍💻 Author

**Patricio Rios Cánepa**

[LinkedIn](https://www.linkedin.com/in/patriciorioscanepa/) | [GitHub](https://github.com/PatricRc) | [Website](https://data-pat-ai.netlify.app/)

## 🙏 Acknowledgements

- [Banco Central de Reserva del Perú](https://www.bcrp.gob.pe/) for providing the API access
- [Streamlit](https://streamlit.io/) for the web app framework
- [OpenAI](https://openai.com/) for AI capabilities 