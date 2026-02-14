import sqlite3
import pandas as pd
import numpy as np
import os
import sys  # 引入sys用于进度条
import warnings
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tabulate import tabulate

warnings.filterwarnings('ignore')

# ================= 1. 系统配置 =================
CONFIG = {
    'db_path': '/sdcard/muben/mokuai/zeus_v400/zeus_data.db',
    'out_dir': '/sdcard/download/sheet',
    'backtest_start': '2024-01-01',
    'live_start': '2026-01-01',    
    'initial_capital': 150000.0,   
    'max_positions': 8,
    'risk_per_trade': 0.04,
    'slippage': 0.0012,
    'fee': 0.0008,
    'capacity_limit': 0.02
}

# ================= 2. 辅助函数：防刷屏进度条 =================
def print_progress(current, total, prefix='Progress', suffix='Complete', decimals=1, length=25, fill='█'):
    """
    调用该函数在终端显示单行进度条
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (current / float(total)))
    filled_length = int(length * current // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    # 使用 \r 回到行首，end='' 不换行
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    if current == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

# ================= 3. 职业因子引擎 =================
def build_factor_score(df):
    # 简单的进度提示
    print("⚡ [1/3] 正在构建核心因子...")
    
    g_code = df.groupby('code')

    df['f_mom'] = g_code['close'].transform(lambda x: x.pct_change(10))
    df['tr'] = np.maximum(df['high'] - df['low'],
                          np.maximum(abs(df['high'] - df['close'].shift(1)),
                                     abs(df['low'] - df['close'].shift(1))))
    df['atr'] = g_code['tr'].transform(lambda x: x.rolling(20).mean())
    df['ma20'] = g_code['close'].transform(lambda x: x.rolling(20).mean())
    df['ma200'] = g_code['close'].transform(lambda x: x.rolling(200).mean())
    df['bias'] = (df['close'] - df['ma20']) / df['ma20']

    v_mean = g_code['amount'].transform(lambda x: x.rolling(20).mean())
    v_std = g_code['amount'].transform(lambda x: x.rolling(20).std())
    df['f_active'] = (df['amount'] - v_mean) / (v_std + 1e-9)

    g_date = df.groupby('date')
    df['final_score'] = g_date['f_mom'].rank(pct=True) * 0.5 + \
                        g_date['f_active'].rank(pct=True) * 0.5

    df.loc[df['bias'] > 0.15, 'final_score'] = 0

    df['is_limit_up'] = df['open'] > df['close'].shift(1) * 1.095
    df['is_limit_down_flat'] = (df['high'] == df['low']) & (df['close'] < df['close'].shift(1) * 0.905)

    return df

# ================= 4. 实盘模拟引擎 =================
class ProV25Engine:
    def __init__(self, df):
        self.df = df[df['date'] >= pd.to_datetime(CONFIG['backtest_start'])]
        self.capital = CONFIG['initial_capital']
        self.positions = {} 
        self.history = []
        self.trades_log = []
        self.live_start_equity = None 

    def run(self):
        dates = sorted(self.df['date'].unique())
        data_by_date = {d: day.set_index('code') for d, day in self.df.groupby('date')}
        
        live_dt = pd.to_datetime(CONFIG['live_start'])
        total_days = len(dates)

        print(f"🚀 [2/3] 开始全周期实战推演 ({total_days}个交易日)...")

        for i, date in enumerate(dates):
            # 调用进度条
            print_progress(i + 1, total_days, prefix='演算进度:', suffix=f'({date.strftime("%Y-%m-%d")})', length=20)
            
            day_df = data_by_date[date]
            
            if self.live_start_equity is None and date >= live_dt:
                curr_val = self.capital
                for c, p in self.positions.items():
                    curr_val += p['shares'] * (day_df.loc[c,'close'] if c in day_df.index else p['entry_p'])
                self.live_start_equity = curr_val

            # --- 卖出管理 ---
            to_sell = []
            for code, pos in self.positions.items():
                if code not in day_df.index: continue
                r = day_df.loc[code]

                pos['stop_loss'] = max(pos['stop_loss'], r['high'] - 2.0 * r['atr'])

                if r['low'] < pos['stop_loss']:
                    if r['is_limit_down_flat']: continue

                    sell_price = pos['stop_loss'] * (1 - CONFIG['slippage'])
                    if r['open'] < pos['stop_loss']: sell_price = r['open'] * (1 - CONFIG['slippage'])

                    val = pos['shares'] * sell_price * (1 - CONFIG['fee'])
                    self.capital += val
                    
                    pnl_amt = val - pos['cost']
                    
                    self.trades_log.append({
                        'P/L': round(pnl_amt, 2)
                    })
                    to_sell.append(code)
            for c in to_sell: del self.positions[c]

            # --- 买入管理 ---
            if len(self.positions) < CONFIG['max_positions']:
                cands = day_df[(day_df['final_score'] > 0) & (day_df['close'] > day_df['ma200'])]
                cands = cands.sort_values('final_score', ascending=False).head(15)

                for code, r in cands.iterrows():
                    if len(self.positions) >= CONFIG['max_positions']: break
                    if code in self.positions or r['is_limit_up']: continue

                    risk_amt = self.capital * CONFIG['risk_per_trade']
                    stop_dist = 2.0 * r['atr'] if r['atr'] > 0 else r['close'] * 0.08
                    shares = (risk_amt / stop_dist // 100) * 100

                    max_shares = (r['amount'] * CONFIG['capacity_limit'] / r['open'] // 100) * 100
                    final_shares = min(shares, max_shares)

                    buy_price = r['open'] * (1 + CONFIG['slippage'])
                    cost = final_shares * buy_price * (1 + CONFIG['fee'])

                    if final_shares >= 100 and self.capital >= cost:
                        self.capital -= cost
                        self.positions[code] = {
                            'shares': final_shares,
                            'entry_p': buy_price,
                            'stop_loss': buy_price - stop_dist,
                            'date': date.strftime('%Y-%m-%d'),
                            'atr': r['atr'],
                            'cost': cost
                        }

            # --- 结算 ---
            total_value = self.capital
            for code, pos in self.positions.items():
                if code in day_df.index:
                    total_value += pos['shares'] * day_df.loc[code, 'close']
                else:
                    total_value += pos['shares'] * pos['entry_p']
            self.history.append({'date': date, 'equity': total_value})

        return pd.DataFrame(self.history).set_index('date'), pd.DataFrame(self.trades_log)

# ================= 5. 高级统计 =================
def calculate_metrics(equity_df, trades_df, live_start_equity):
    equity_df['returns'] = equity_df['equity'].pct_change()
    total_ret = (equity_df['equity'].iloc[-1] / CONFIG['initial_capital'] - 1) * 100
    
    running_max = equity_df['equity'].cummax()
    drawdown = (equity_df['equity'] - running_max) / running_max * 100
    max_dd = drawdown.min()
    
    live_start_val = live_start_equity if live_start_equity else CONFIG['initial_capital']
    live_ret = (equity_df['equity'].iloc[-1] / live_start_val - 1) * 100
    
    live_df = equity_df[equity_df.index >= pd.to_datetime(CONFIG['live_start'])]
    if not live_df.empty:
        l_peak = live_df['equity'].cummax()
        l_dd = ((live_df['equity'] - l_peak) / l_peak).min() * 100
    else:
        l_dd = 0.0

    volatility = equity_df['returns'].std() * np.sqrt(250) * 100
    sharpe = (equity_df['returns'].mean() * 250 - 0.03) / (equity_df['returns'].std() * np.sqrt(250) + 1e-9)

    if not trades_df.empty:
        win_trades = trades_df[trades_df['P/L'] > 0]
        loss_trades = trades_df[trades_df['P/L'] <= 0]
        
        win_rate = len(win_trades) / len(trades_df) * 100
        avg_win = win_trades['P/L'].mean() if not win_trades.empty else 0
        avg_loss = abs(loss_trades['P/L'].mean()) if not loss_trades.empty else 0
        pl_ratio = avg_win / avg_loss if avg_loss != 0 else 0
        
        kelly = (pl_ratio * (win_rate/100) - (1 - win_rate/100)) / pl_ratio if pl_ratio > 0 else 0
    else:
        win_rate, pl_ratio, kelly = 0, 0, 0

    return {
        '总收益率 (回测)': f"{total_ret:.2f}%",
        '最大回撤 (回测)': f"{max_dd:.2f}%",
        '实盘收益 (1月起)': f"{live_ret:.2f}%", 
        '实盘回撤 (1月起)': f"{l_dd:.2f}%",
        '年化波动率': f"{volatility:.2f}%",
        '夏普比率': f"{sharpe:.2f}",
        '胜率': f"{win_rate:.2f}%",
        '盈亏比': f"{pl_ratio:.2f}",
        '凯利仓位': f"{kelly:.2f}",
        '交易总数': len(trades_df)
    }

# ================= 6. 终端报表输出 =================
def print_reports(metrics, positions, candidates, latest_prices, total_capital):
    print("\n" + "="*50)
    print(f"⚡ ZEUS PRO V3.4 指挥官实战终端")
    print("="*50)
    
    # 1. 核心指标
    metric_list = [[k, v] for k, v in metrics.items()]
    print("\n📊 [核心绩效指标]")
    print(tabulate(metric_list, headers=["指标", "数值"], tablefmt="fancy_grid"))

    # 2. 持仓监控
    pos_data = []
    sell_alert = [] 
    
    total_mkt_val = 0
    for code, pos in positions.items():
        curr = latest_prices.get(code, pos['entry_p'])
        total_mkt_val += pos['shares'] * curr

    for code, pos in positions.items():
        shares = pos['shares']
        cost = pos['cost'] / shares
        curr = latest_prices.get(code, cost)
        pnl = (curr/cost - 1) * 100
        risk_dist = (curr - pos['stop_loss']) / curr * 100
        
        action = "持有"
        if risk_dist < 0:
            action = "!! 立即卖出 !!"
            # 🚀 增强版卖出预警：加入持股数
            sell_alert.append([code, shares, curr, pos['stop_loss'], "已跌破止损"])
        elif risk_dist < 2.0:
            action = "! 预警 !"
            sell_alert.append([code, shares, curr, pos['stop_loss'], "接近止损(<2%)"])
        
        pos_data.append([
            code, shares, f"{cost:.2f}", f"{curr:.2f}", 
            f"{pnl:+.2f}%", f"{pos['stop_loss']:.2f}", 
            f"{risk_dist:.1f}%", action
        ])
    
    print(f"\n🛡️ [实盘持仓监控] (当前仓位: {total_mkt_val/total_capital*100:.1f}%)")
    if pos_data:
        headers = ["代码", "持股", "成本", "现价", "盈亏", "止损价", "安全垫", "指令"]
        print(tabulate(pos_data, headers=headers, tablefmt="simple"))
    else:
        print("  >> 当前空仓")

    # 3. 卖出/预警指令 (🚀 新增列：持股)
    if sell_alert:
        print("\n🚨 [卖出/风控预警] (重点关注)")
        headers_sell = ["代码", "持股", "现价", "止损价", "触发原因"]
        print(tabulate(sell_alert, headers=headers_sell, tablefmt="grid"))
    else:
        print("\n✅ [风控扫描] 无触发止损标的")

    # 4. 买入建议
    buy_data = []
    print(f"\n🎯 [明日开仓指令 (Top 5)]")
    print(f"   ⚠️ 已强制按实盘本金 [{CONFIG['initial_capital']:.0f}元] 测算股数")
    
    real_capital = CONFIG['initial_capital'] 
    
    for c in candidates:
        risk_amt = real_capital * CONFIG['risk_per_trade']
        stop_dist = c['price'] - c['stop']
        if stop_dist <= 0: stop_dist = c['price'] * 0.02
        
        shares = int((risk_amt / stop_dist // 100) * 100)
        max_shares_cap = (real_capital * 0.2) / c['price'] // 100 * 100
        shares = min(shares, max_shares_cap)
        
        est_val = shares * c['price']
        
        if shares > 0:
            buy_data.append([
                c['code'], c['price'], shares, 
                f"{est_val:.0f}", f"{c['stop']:.2f}"
            ])
            
    if buy_data:
        headers = ["代码", "触发价", "建议股数", "预估金额", "初始止损"]
        print(tabulate(buy_data, headers=headers, tablefmt="grid"))
    else:
        print("  >> 无符合条件的开仓标的")
    print("="*50 + "\n")

# ================= 7. 主程序 =================
if __name__ == "__main__":
    conn = sqlite3.connect(CONFIG['db_path'])
    df = pd.read_sql("SELECT * FROM daily_data WHERE date >= '2023-01-01'", conn)
    conn.close()
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['code', 'date'])
    df = build_factor_score(df)

    engine = ProV25Engine(df)
    equity, trades = engine.run()

    # 指标计算
    metrics = calculate_metrics(equity, trades, engine.live_start_equity)
    
    latest_date = equity.index[-1]
    latest_prices = df[df['date'] == latest_date].set_index('code')['close'].to_dict()
    final_capital = equity['equity'].iloc[-1]
    
    # 选股
    day_df = df[df['date'] == latest_date].set_index('code')
    cands = day_df[(day_df['final_score'] > 0) & (~day_df.index.isin(engine.positions.keys()))]
    cands = cands.sort_values('final_score', ascending=False).head(5)
    
    buy_cands = []
    for code, row in cands.iterrows():
        stop_dist = 2.0 * row['atr']
        buy_cands.append({
            'code': code, 'price': row['close'], 
            'stop': row['close'] - stop_dist
        })

    # 输出
    os.makedirs(CONFIG['out_dir'], exist_ok=True)
    trades.to_excel(os.path.join(CONFIG['out_dir'], 'Zeus_Trades.xlsx'), index=False)
    
    print_reports(metrics, engine.positions, buy_cands, latest_prices, final_capital)
