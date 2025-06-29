from config.base_config import SEED
from config.env_config import ENV_SETTINGS
from config.model_config import PPO_CONFIG
from data.fetch_data import fetch_yfinance_data
from data.preprocess import engineer_features
from data.split_data import train_test_split
from envs.trading_env import TradingEnv
from models.train_agent import train_agent
from models.evaluate_agent import evaluate_agent
from utils.logger import setup_logger
from utils.plotter import plot_equity_curve, plot_drawdown

def main():
    logger = setup_logger()
    logger.info("Fetching data...")
    df = fetch_yfinance_data(['AAPL', 'GOOG'], start='2018-01-01', end='2024-01-01')
    logger.info("Engineering features...")
    features = engineer_features(df)
    logger.info("Splitting data...")
    train_df, test_df = train_test_split(features)
    logger.info("Initializing environment...")
    env = TradingEnv(train_df, **ENV_SETTINGS)
    logger.info("Training agent...")
    agent = train_agent(env, PPO_CONFIG)
    logger.info("Evaluating agent...")
    test_env = TradingEnv(test_df, **ENV_SETTINGS)
    results = evaluate_agent(agent, test_env)
    logger.info(f"Test results: {results}")
    plot_equity_curve(results, title="Test Equity Curve")

if __name__ == "__main__":
    main()
