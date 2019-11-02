import bitstamp.client

trading_client = bitstamp.client.Trading()

ammount = 0.01
trading_client.buy_market_order(ammount=ammount, base="btc", quote="eur")
