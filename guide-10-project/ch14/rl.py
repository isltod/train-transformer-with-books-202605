import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yfinance as yf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import random
import pandas as pd


class StockTradingEnv(gym.Env):
    # 강화 학습 환경에 화면에 직접 시각적 렌더링을 지원한다는 것을 알려주는 메타데이터 설정
    metadata = {"render_modes": ["human"]}

    # 원본 주가, 일별 백분율 변화 데이터프레임 받고...
    def __init__(self, df, pct_df, max_steps=1000, render_mode=None):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.pct_df = pct_df
        self.render_mode = render_mode
        self.reward_range = (-np.inf, np.inf)
        # 선택할 수 있는 행동 옵션은...관측 공간은 몇일 이동 평균인지에서 그 윈도우인데...
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5, 5), dtype=np.float16
        )
        # 초기 자본금은 만...왜 다들 100을 안 쓰는지...
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.shares_held = 0
        # 현재 스텝이 6에서 시작?
        self.current_step = 6
        self.max_steps = max_steps
        # 그냥 len(df)해도 마찬가진데...왜 굳이 loc["close"]를...
        # 거기서 6 빼는건 위와 맞춤이라 그렇다 치고, 2는 또 왜 빼나?
        self.train_cnt_epoch = len(self.df.loc[:, "Close"].values) - 2 - 6

    def step(self, action):
        # 아마도 지금까지 몇 번 돌았나를 기록에 남기는 듯...
        self.current_step += 1
        # 느낌상 매수/매도는 action[0], 주문 수량은 action[1]?
        action_type = action[0]
        amount = action[1]

        # 오늘과 내일 종가를 받아서...
        # 원저자 코드에 다음 두 줄 뒤에 item()을 추가하여 에러 방지
        close_price = self.df.loc[self.current_step, "Close"].item()
        next_day_close_price = self.df.loc[self.current_step + 1, "Close"].item()

        # 이 둘은 인스턴스나 클래스 변수가 아니라 이렇게 초기화하지 않아도 될거 같은데...
        shares_bought = 0
        shares_sold = 0

        # 거래 시도 전 평가 자산 - 현금 + 주식 수 * 현재 가격
        asset_value_before_action = self.balance + self.shares_held * close_price

        # 1은 보류, 2는 매수, 3은 청산인가...
        if action_type < 1:
            # 보유(Hold)
            pass
        elif action_type < 2:
            # 매수 - 현재 현금으로 가능한 수량 산걸로...주식은 늘리고 현금은 줄이고...
            total_possible = int(self.balance / close_price)
            shares_bought = int(total_possible * amount)
            total_cost = shares_bought * close_price
            self.balance -= total_cost
            self.shares_held += shares_bought
        elif action_type < 3:
            # 매도 = 현재 가격으로 매도 금액 계산해서 현금 늘리고 주식은 줄이고...
            shares_sold = int(self.shares_held * amount)
            self.balance += shares_sold * close_price
            self.shares_held -= shares_sold

        # 아마도 로직상 데이터 끝인 모양...각 에포크 당 데이터 수 넘어가면 초기화...
        # 여기 6이 하드코딩인게 계속 걸리네...
        if self.current_step >= self.train_cnt_epoch:
            self.current_step = 6

        obs = self._next_observation()

        # 거래 시도 후 평가 자산
        asset_value_after_action = (
            self.balance + self.shares_held * next_day_close_price
        )
        # 보상은 거래 시도 전후 평가 자산 차이로...
        reward = asset_value_after_action - asset_value_before_action

        # Gymnasium 규격에 맞춰 terminated와 truncated를 구분하여 반환
        terminated = self.balance <= 0
        truncated = self.current_step >= self.max_steps

        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 초기화는 자산, 주식, 현재 포커스(6일) 설정...
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = 6
        obs = self._next_observation()
        return obs, {}

    def _next_observation(self):
        end_slice = self.current_step + 1
        start_slice = end_slice - 4
        obs = self.pct_df.iloc[start_slice : end_slice + 1].values
        return obs.astype(np.float16)

    def render(self):
        return self.df.loc[self.current_step, "Open"]

    def close(self):
        return


def get_yf_data(symbol, start_date, end_date):
    df = yf.download(symbol, start_date, end_date)

    # yfinance에서 반환한 DataFrame의 컬럼이 MultiIndex 형태인 경우 평탄화(Flat) 처리
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    # observation_space shape인 (5, 5)에 대응하도록 5개의 핵심 컬럼만 선택하여 고정
    df = df[["Close", "High", "Low", "Open", "Volume"]]
    df = df.sort_index()

    # 여기서 날짜 인덱스는 없어지고...
    df = df.reset_index(drop=True)
    # 새 df 복사해서 만들고...일간 백분율 변동으로 바꾸기...
    pct_df = df.copy(deep=True)
    pct_df = pct_df.pct_change()
    print(df.head(20))
    print(pct_df.head(20))

    return df, pct_df


# 훈련 데이터는 애플 2020~2022
df, pct_df = get_yf_data("AAPL", "2020-01-01", "2023-01-01")

# DummyVecEnv 클래스는 환경에 대한 벡터화된 래퍼를 생성
env = DummyVecEnv([lambda: StockTradingEnv(df, pct_df)])

model = PPO("MlpPolicy", env, verbose=1, device="cpu")
model.learn(total_timesteps=10000)

md_path = "../../data/ppo_stock"
model.save(md_path)


# 추론에는 이 클래스를 쓴다는데...위에 StockTradingEnv 클래스와 거의 같은 코드
class StockTradingTestEnv(StockTradingEnv):
    """
    여기에 이 주석을 달아야 설명에 뜨나?
    추론에는 이 클래스를 쓴다는데...위에 StockTradingEnv 클래스와 거의 같은 코드
    """

    def __init__(self, df, pct_df, initial_balance=10000):
        super().__init__(df, pct_df)
        self.initial_balance = initial_balance
        self.balance = self.initial_balance
        self.train_cnt_epoch = len(self.df.loc[:, "Close"].values) - 2 - 6

    def step(self, action):
        self.current_step += 1
        action_type = action[0]
        amount = action[1]

        # 원저자 코드에 다음 두 줄 뒤에 item()을 추가하여 에러 방지
        close_price = self.df.loc[self.current_step, "Close"].item()
        next_day_close_price = self.df.loc[self.current_step + 1, "Close"].item()

        shares_bought = 0
        shares_sold = 0
        asset_value_before_action = self.balance + self.shares_held * close_price

        if action_type < 1:
            # 보유(Hold)
            pass
        elif action_type < 2:
            # 구매(Buy)
            total_possible = int(self.balance / close_price)
            shares_bought = int(total_possible * amount)
            total_cost = shares_bought * close_price
            self.balance -= total_cost
            self.shares_held += shares_bought
        elif action_type < 3:
            # 판매(Sell)
            shares_sold = int(self.shares_held * amount)
            self.balance += shares_sold * close_price
            self.shares_held -= shares_sold

        if self.current_step >= len(self.df.loc[:, "Close"].values) - 6:
            self.current_step = 6

        obs = self._next_observation()

        asset_value_after_action = (
            self.balance + self.shares_held * next_day_close_price
        )
        reward = asset_value_after_action - asset_value_before_action

        # 이 부분만 다른데...자본금 0 조건은 없고, max_step 대신 train_cnt_epoch 설정 만큼만 돌아가도록...
        # Gymnasium 규격에 맞춰 terminated와 truncated를 구분하여 반환
        if self.current_step >= self.train_cnt_epoch:
            terminated = True
            truncated = True
        else:
            terminated = False
            truncated = False

        return obs, reward, terminated, truncated, {}


import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# 저장된 모델 불러오기
model = PPO.load(md_path)

# 추론이라기보다는 새 데이터셋으로 시험하는 모양
df_2023, pct_df_2023 = get_yf_data("AAPL", "2023-01-01", "2023-05-30")

# 새 데이터용 환경 설정 및 (학습에서 얻은) final balance 조정
final_training_balance = 100000
env = DummyVecEnv(
    [
        lambda: StockTradingTestEnv(
            df_2023, pct_df_2023, initial_balance=final_training_balance
        )
    ]
)

# 환경의 초기 상태 설정
state = env.reset()
done = False

# 이 리스트는 각 스텝(step)에서 포트폴리오의 값을 저장
portfolio_values = []

while not done:
    # 모델(model)로부터 action 구하기
    action, _ = model.predict(state)
    # print('printing action')
    # print(action)

    # 환경에서 첫 스텝을 실행하고 새로운 상태(state)와 보상(reward) 구하기
    state, reward, done, info = env.step(action)

    if not done:
        # 이게 현재 평가 자산인 듯...
        portfolio_value = env.envs[0].balance + (
            env.envs[0].shares_held
            * env.envs[0].df.loc[env.envs[0].current_step, "Close"]
        )
        # print('balance', .env.envs[0].balance)
        # print('shares_held', .env.envs[0].shares_held)
        # print('portfolio_value', portfolio_value)
        # 포트폴리오 값을 리스트에 추가
        portfolio_values.append(portfolio_value)
        # print('portfolio_values', portfolio_values)
        # print('current_step', .env.envs[0].current_step)
    else:
        print("Reached the end of the data.")


# 시간에 걸친 포트폴리오 값 디스플레이
plt.figure(figsize=(10, 6))
plt.plot(portfolio_values)
plt.title("Portfolio Value Over Time")
plt.xlabel("Step")
plt.ylabel("Value")
plt.show()
# 그림을 보니 이상하게 잘 딴다...
