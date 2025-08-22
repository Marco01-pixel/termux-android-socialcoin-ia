from flask import Flask
import socialcoin_ai  # tu módulo principal

app = Flask(__name__)

@app.route("/")
def home():
    return "🚀 SocialCoin IA funcionando correctamente"

# Aquí puedes exponer funciones de socialcoin_ai a través de rutas si quieres
