from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from datetime import datetime 
from alpaca_trade_api import REST 
from timedelta import Timedelta 
from finbert_utils import estimate_sentiment

API_KEY = "PKOG15H42CI1QEWVG3YZ" 
API_SECRET = "kYCv76qQzJX9FtSPd8pXBYqhi3hL9Ryb7QZILoVC" 
BASE_URL = "https://paper-api.alpaca.markets/v2"

ALPACA_CREDS = {
    "API_KEY":API_KEY, 
    "API_SECRET": API_SECRET, 
    "PAPER": True
}


# trading logic
class MLTrader(Strategy): 
    def initialize(self, symbol:str="SPY", cash_at_risk:float=.05): 
        self.symbol = symbol # what stock to trade
        self.sleeptime = "24H" # how frequently we trade
        self.last_trade = None 
        self.cash_at_risk = cash_at_risk
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)

    def position_sizing(self): 
        cash = self.get_cash() # how much cash in account
        last_price = self.get_last_price(self.symbol) 
        quantity = round(cash * self.cash_at_risk / last_price, 0) # how many shares to buy
        return cash, last_price, quantity

    def get_dates(self): 
        today = self.get_datetime()
        three_days_prior = today - Timedelta(days=3)
        return today.strftime('%Y-%m-%d'), three_days_prior.strftime('%Y-%m-%d')

    def get_sentiment(self): 
        # get sentiment from news for past three days
        today, three_days_prior = self.get_dates()
        news = self.api.get_news(symbol=self.symbol, 
                                 start=three_days_prior, 
                                 end=today) 
        news = [ev.__dict__["_raw"]["headline"] for ev in news]
        probability, sentiment = estimate_sentiment(news)
        return probability, sentiment 

    # runs every time we get new data
    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing() 
        probability, sentiment = self.get_sentiment()

        if cash > last_price: # can't buy if you don't have enough cash
            if sentiment == "positive" and probability > .7: 
                if self.last_trade == "sell": 
                    self.sell_all() 
                order = self.create_order( # create order
                    self.symbol, 
                    quantity, 
                    "buy", 
                    type="bracket", 
                    take_profit_price=last_price*1.10, 
                    stop_loss_price=last_price*.90
                )
                self.submit_order(order) # make order
                self.last_trade = "buy"
            elif sentiment == "negative" and probability > .7: 
                if self.last_trade == "buy": 
                    self.sell_all() 
                order = self.create_order( # create order
                    self.symbol, 
                    quantity, 
                    "sell", 
                    type="bracket", 
                    take_profit_price=last_price*.90, 
                    stop_loss_price=last_price*1.10
                )
                self.submit_order(order) # make order
                self.last_trade = "sell"

start_date = datetime(2024,1,1)
end_date = datetime(2024,10,1) 
broker = Alpaca(ALPACA_CREDS) 

# create instance of MLTrader class
strategy = MLTrader(name='mlstrat', broker=broker, 
                    parameters={"symbol":"SPY", 
                                "cash_at_risk":.5})

# evaluate how well the bot runs
strategy.backtest(
    YahooDataBacktesting, 
    start_date, 
    end_date, 
    parameters={"symbol":"SPY", "cash_at_risk":.5}
)
# trader = Trader()
# trader.add_strategy(strategy)
# trader.run_all()