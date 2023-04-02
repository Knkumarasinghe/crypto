import pickle, json
import numpy as np
import tensorflow as tf
from flask_cors import CORS
from datetime import datetime, timedelta
from yahoo_fin.stock_info import get_data
from flask import Flask, Response, request

btc_model = tf.keras.models.load_model('weights/model_wf BTC.h5')
eth_model = tf.keras.models.load_model('weights/model_wf ETH.h5')
usdt_model = tf.keras.models.load_model('weights/model_wf USDT.h5')

btc_scaler = pickle.load(open('weights/scaler BTC.pkl', 'rb'))
eth_scaler = pickle.load(open('weights/scaler ETH.pkl', 'rb'))
usdt_scaler = pickle.load(open('weights/scaler USDT.pkl', 'rb'))

app = Flask(__name__)
CORS(app)

def get_value(COIN):
    today = datetime.today().strftime('%m/%d/%Y')
    today_minus_30_days = (datetime.today() - timedelta(days=30)).strftime('%m/%d/%Y')

    df_coin = get_data(f"{COIN}-USD", start_date=today_minus_30_days, end_date=today, index_as_date = True, interval="1d")
    df_coin = df_coin[['open','high','low','close']]
    df_coin = df_coin.dropna()
    df_coin = df_coin.tail(15)
    
    Xinf = df_coin.values
    prediction_dummy = np.zeros((1, 4))
    if COIN == 'BTC':
        Xinf = btc_scaler.transform(Xinf).reshape(1,15,4)
        prediction = btc_model.predict(Xinf)
        prediction_dummy[0,-1] = prediction[0,-1]
        prediction = btc_scaler.inverse_transform(prediction_dummy)
        prediction = prediction[0,-1]

    elif COIN == 'ETH':
        Xinf = eth_scaler.transform(Xinf).reshape(1,15,4)
        prediction = eth_model.predict(Xinf)
        prediction_dummy[0,-1] = prediction[0,-1]
        prediction = eth_scaler.inverse_transform(prediction_dummy)
        prediction = prediction[0,-1]

    elif COIN == 'USDT':
        Xinf = usdt_scaler.transform(Xinf).reshape(1,15,4)
        prediction = usdt_model.predict(Xinf)
        prediction_dummy[0,-1] = prediction[0,-1]
        prediction = usdt_scaler.inverse_transform(prediction_dummy)
        prediction = prediction[0,-1]

    return prediction


@app.route("/forecast", methods=["POST"])
def forecast():
    '''
        {
            "COIN" : "USDT"
        }
    '''
    COIN = request.json['COIN']
    value = get_value(COIN)

    return Response(
                    response=json.dumps({
                                    "value": f"{value}"
                                    }), 
                    mimetype="application/json", 
                    status=200
                    )

if __name__ == "__main__": 
    app.run(
            debug=True, 
            host='0.0.0.0', 
            port=5000
            )