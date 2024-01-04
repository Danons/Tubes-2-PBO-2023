import pandas as pd
import numpy as np
import csv

import mysql.connector
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import time
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from matplotlib.widgets import RadioButtons
from matplotlib.animation import FuncAnimation
from datetime import date
from bs4 import BeautifulSoup
from tabulate import tabulate
from tqdm import tqdm
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from colorama import Fore,Style


datadaerahrig = {
    'Daerah': ['Amerika Serikat', 'Kanada', 'Eropa', 'Asia', 'Negara lainnya']
}

datajmlhrig = {
    'Jumlah Rig': [53, 60, 17, 35, 12]
}

datahasilminyak = {
    'No': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Negara_Produsen': ['Arab Saudi', 'Rusia', 'Amerika Serikat', 'Iran', 'Tiongkok', 'Kanada', 'Meksiko', 'Uni Emirat Arab', 'Kuwait', 'Venezuela'],
    'Barel(2006)': [10.665, 9.677, 8.331, 4.148, 3.846, 3.288, 3.707, 2.945, 2.675, 2.803],
    'Barel(2007)': [10.234, 9.876, 8.481, 4.043, 3.901, 3.358, 3.501, 2.948, 2.613, 2.667],
    'Pangsa_Pasar': [0.118, 0.12, 0.111, 0.51, 0.48, 0.40, 0.36, 0.34, 0.30, 0.30]
}

datapegawai = {
    'No': [1, 2, 3, 4, 5],
    'Nama': ['Hafidz', 'Sapi', 'Ayu', 'Salsabila', 'Niken'],
    'Target_awal': [9, 10, 16, 30, 18],
    'Total': [0, 0, 0, 0, 0],
    'Tanggal_Penghitungan': [None, None, None, None, None] 
}

df = pd.DataFrame(datahasilminyak)
df_pegawai = pd.DataFrame(datapegawai)

class MySQL():
    def TampilMatplotlibRigAnimated(self):
        fig,ax =  plt.subplots()
        
        def init():
            ax.bar([], [])
            ax.set_xlabel('Negara')
            ax.set_ylabel('Jumlah Rig')
            ax.set_title('Jumlah Rig per Benua')
            return ax
        
        def update(frame):
            ax.clear()
            ax.bar(datadaerahrig['Daerah'][:frame + 1], datajmlhrig['Jumlah Rig'][:frame + 1])
            ax.set_xlabel('Negara')
            ax.set_ylabel('Jumlah Rig')
            ax.set_title('Jumlah Rig per Benua')
            plt.pause(0.5)
        
        frames = len(datadaerahrig['Daerah'])
        animation = FuncAnimation(fig, update, frames=frames, init_func=init, blit=False, repeat=False)
        plt.show()
        
    def TampilMatplotlibOilProductionAnimated(self,df):
        fig, ax = plt.subplots()

        bar1 = ax.bar(df['Negara_Produsen'], df['Barel(2006)'], label='Barel(2006)', alpha=1)
        bar2 = ax.bar(df['Negara_Produsen'], df['Barel(2007)'], bottom=df['Barel(2006)'], label='Barel(2007)', alpha=1)

        ax.set_xlabel('Negara Produsen')
        ax.set_ylabel('Barel')
        ax.legend(loc='upper right')

        plt.xlim(0,10)
        plt.ylim(0,30)
        
        plt.show()
    
    def scrape_article(self):
        url = 'https://ekonomi.kompas.com/read/2018/05/22/141550826/prospek-saham-emiten-perminyakan-menguat'
        response = requests.get(url)

        if response.status_code == 200:
            time.sleep(3)
            soup = BeautifulSoup(response.text, 'html.parser')

            article_content = soup.find('div', class_='read__content')

            if article_content:
                article_text = article_content.get_text()
                print('')
                print('Berikut adalah artikel secara utuh : ')
                print(article_text)
                print('')
            else:
                print("Tidak dapat menemukan konten artikel.")

            parser = PlaintextParser.from_string(article_text, Tokenizer('English'))
            summarizer = LsaSummarizer()
            summary = summarizer(parser.document, sentences_count=5)
            print('')
            print('Berikut adalah summary dari artikel :')
            print('')
            for sentence in summary:
                print(sentence)
            print('')

            target_words = ['menanjak', 'mengerek', 'menembus', 'tertinggi', 'kenaikan', 'bisa naik', 'naik']
            sia = SentimentIntensityAnalyzer()

            print('')
            print("Analisis sentimen terhadap summary dari artikel :")
            print('')

            aggregated_sentiment_score = 0.0
            total_sentences = 0

            for sentence in summary:
                highlighted_sentence, sentiment_score = self.highlight_word_and_adjust_sentiment(sentence, target_words, sia)
                print(f'Original Sentence: {sentence}')
                print(f'Highlighted Sentence: {highlighted_sentence}')
                print(f'Sentiment Score: {sentiment_score}\n')

                aggregated_sentiment_score += sentiment_score
                total_sentences += 1

            average_sentiment_score = aggregated_sentiment_score / total_sentences

            if average_sentiment_score >= 0.5:
                overall_sentiment = 'Positive'
            elif average_sentiment_score <= -0.1:
                overall_sentiment = 'Negative'
            else:
                overall_sentiment = 'Neutral'

            print(f'Overall Sentiment: {overall_sentiment}')

    def highlight_word_and_adjust_sentiment(self, sentence, target_words, sia):
        words = word_tokenize(str(sentence))  

        highlighted_sentence = ''
        sentiment_adjustment = 0.0

        for word in words:
            if word.lower() in target_words:
                highlighted_sentence += Fore.RED + word + Style.RESET_ALL + ' '
                sentiment_adjustment -= 1  
            else:
                highlighted_sentence += word + ' '

        sentiment_score = sia.polarity_scores(' '.join(words))['compound'] + sentiment_adjustment

        return highlighted_sentence.strip(), sentiment_score

    def TampilDataFrame(self):
        df = pd.read_csv('migas.csv')
        fig, ax = plt.subplots()
        df['2006'] = df['2006']*-1

        def animate(tahun):
            ax.clear()
            fil = df[df['Tahun']==tahun]
            y1=plt.barh(y=fil['Negara produsen'], width=fil['2006'])
            y2=plt.barh(y=fil['Negara produsen'], width=fil['2007'])

            ax.set_xlim(-15000,15000)
            ax.bar_label(y1, padding=3, labels=[f'{-1*round(value, -1):,}' for value in fil['2006']])
            ax.bar_label(y2, padding=3, labels=[f'{round(value, -1):,}' for value in fil['2007']])

            for edge in ['top', 'right', 'bottom', 'left']:
                ax.spines[edge].set_visible(False)

            ax.tick_params(left=False)
            ax.get_xaxis().set_visible(False)

            ax.legend([y1, y2], ['Kuartal 1', 'Kuartal 2'], loc='upper right')

            ax.set_title(f'Penghasilan barel(juta) minyak per tahun {tahun}', size=18, weight='bold')
            plt.pause(1)

        frames = range(df['Tahun'].min(), df['Tahun'].max()+1)
        animation = FuncAnimation(fig, animate, frames=frames, interval=1000)
        plt.show()


    def CsvKeXAMPP(self):
        try:
            connection = mysql.connector.connect(
                host="localhost",
                user="root",
                password="",
                database="dataperusahaan"
            )
            cursor = connection.cursor()

            check_query = "SELECT COUNT(*) FROM datahasilminyak"
            cursor.execute(check_query)
            count = cursor.fetchone()[0]

            if count > 0:
                print("Data sudah ada, tolong jangan gunakan menu ini")
                return

            create_table_query = """
            CREATE TABLE IF NOT EXISTS datahasilminyak (
                `No` INT PRIMARY KEY,
                `Negara_Produsen` VARCHAR(255),
                `Barel_2006` FLOAT,
                `Barel_2007` FLOAT,
                `Pangsa_Pasar` FLOAT
            )
            """
            cursor.execute(create_table_query)

            for index, row in enumerate(datahasilminyak['No']):
                insert_query = f"""
                INSERT INTO datahasilminyak (`No`, `Negara_Produsen`, `Barel_2006`, `Barel_2007`, `Pangsa_Pasar`)
                VALUES ({row}, '{datahasilminyak['Negara_Produsen'][index]}', {datahasilminyak['Barel(2006)'][index]}, 
                        {datahasilminyak['Barel(2007)'][index]}, {datahasilminyak['Pangsa_Pasar'][index]})
                """
                cursor.execute(insert_query)

            connection.commit()

            print("Data berhasil dimasukkan lewat Database MYSQL")

        except mysql.connector.Error as err:
            if err.errno == mysql.connector.errorcode.ER_ACCESS_DENIED_ERROR:
                print("Error: Access denied. Check your username and password.")
            elif err.errno == mysql.connector.errorcode.ER_BAD_DB_ERROR:
                print("Error: Database does not exist.")
            else:
                print(f"Error: {err}")
        finally:
            if 'connection' in locals() and connection.is_connected():
                cursor.close()
                connection.close()
    
    def HeatmapDataHasilMinyak(self,data):
        columns_for_heatmap = ['Barel(2006)', 'Barel(2007)']

        heatmap_data = data[columns_for_heatmap]

        plt.figure(figsize=(10, 6))
        sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt='.2f', linewidths=.5)
        plt.title('Heatmap Data Hasil Minyak')
        plt.show()

class WebScrapping():
    def __init__(self, nama, nim, kelas):
        self.nama = nama
        self.nim = nim
        self.kelas = kelas

    def User(self):
        print(f'Nama  : {self.nama}')
        print(f'NIM   : {self.nim}')
        print(f'Kelas : {self.kelas}')

    def Stop(self):
        print(f'===== Terimakasih Telah Mengakses Perusahaan Migas Kami 😊! =====')
    
    def MencariDataMinyak(self):
        URL = 'https://id.wikipedia.org/wiki/Minyak_bumi'
        response = requests.get(URL)
        soup = BeautifulSoup(response.text, 'html.parser')

        table = soup.find('table', {'class': 'wikitable sortable'}).tbody
        rows = table.find_all('tr')
        columns = [v.text.replace('\n', '') for v in rows[0].find_all('th')]

        df = pd.DataFrame(columns=columns)

        for i in range(1, len(rows)):
            tds = rows[i].find_all('td')

            if len(tds) == 4:
                values = [tds[0].text, tds[1].text, tds[2].text, tds[3].text.replace('\n', '').replace('\xa0', '')]
            else:
                values = [td.text.replace('\n', '').replace('\xa0', '') for td in tds]

            df = df._append(pd.Series(values, index=columns), ignore_index=True)

        df.to_csv('scraped_data.csv', index=False)
        df_from_csv = pd.read_csv('scraped_data.csv')
        print(df_from_csv)

    def PengertianMinyakBumi(self):
        url = 'https://id.wikipedia.org/wiki/Minyak_bumi'
        response = requests.get(url)

        def wikibot(url):
            url_open = requests.get(url)
            soup = BeautifulSoup(url_open.content, 'html.parser')
            details = soup('table', {'class': 'infobox'})
            for i in details:
                h = i.find_all('tr')
                for j in h:
                    heading = j.find_all('th')
                    detail = j.find_all('td')
                    if heading is not None and detail is not None:
                        for x, y in zip(heading, detail):
                            print("{}  :  {}".format(x.text, y.text))
                            print("----------")
            for z in range(1, 3):
                print(soup('p')[z].text)
        wikibot(url)

if __name__ == "__main__":
    aksesmysql = MySQL()
    akseswbscrap = WebScrapping('Muhammad Hafidz Darul Quro', '1103223052', 'TK-46-05')
    akseswbscrap.User()

    while True:
        print('')
        print('========                    MENU                       ========')
        print('')
        print('1. Pengertian Minyak Bumi')
        print('2. Memasukkan data ke dalam xampp')
        print('3. Tampil data grafik')
        print('4. WebScrapping langsung dari wikipedia')
        print('5. Webscrapping artikel migas, Terdapat berita baru tentang lonjakan harga minyak')
        print('6. Tampil data Heatmap')
        print('7. Tampil data hasil minyak dari 40 negara')
        print('0. Stop Menu')
        print('')
        inputmenu = int(input("Masukkan Pilihan yang anda inginkan : "))
        print('')

        if inputmenu == 1:
            akseswbscrap.PengertianMinyakBumi()
        elif inputmenu == 2:
            aksesmysql.CsvKeXAMPP()
        elif inputmenu == 3:
            print('')
            print('1. Tampil data rig')
            print("2. Tampil data hasil minyak per benua")
            print('')
            inputmenu3 = int(input('Masukkan Pilihan yang anda inginkan : '))
            if inputmenu3 == 1:
                aksesmysql.TampilMatplotlibRigAnimated()
            elif inputmenu3 == 2:
                aksesmysql.TampilMatplotlibOilProductionAnimated(df)
        elif inputmenu == 4 :
            akseswbscrap.MencariDataMinyak()
        elif inputmenu == 5 :
            aksesmysql.scrape_article()
        elif inputmenu == 6 :
            aksesmysql.HeatmapDataHasilMinyak(df)
        elif inputmenu == 7 :
            aksesmysql.TampilDataFrame()
        elif inputmenu == 0:
            akseswbscrap.Stop()
            break