import sqlite3
import pandas as pd
import numpy as np
import time
import os
import warnings
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ================= 1. 系统配置 (V2.5 极致实战修复) =================
CONFIG = {
    'db_path': '/sdcard/muben/mokuai/zeus_v400/zeus_data.db',
    'out_dir': '/sdcard/download/sheet',
    'backtest_start': '2024-01-01', 
    'initial_capital': 150000.0,
    'max_positions': 8,
    'risk_per_trade': 0.04,         # 还原 V2.2 的高攻击力
    'slippage': 0.0012,             # 保持严格滑点
    'fee': 0.0008,
    'capacity_limit': 0.02          
}

# ================= 2. 职业因子引擎 (截面排序) =================
def build_factor_score(df):
    print("🚀 [1/3] 正在重构因子引擎：引入每日截面排序，消除未来函数...")
    g_code = df.groupby('code')
    
    # 基础因子
    df['f_mom'] = g_code['close'].transform(lambda x: x.pct_change(10))
    df['tr'] = np.maximum(df['high'] - df['low'], 
               np.maximum(abs(df['high'] - df['close'].shift(1)), 
                          abs(df['low'] - df['close'].shift(1))))
    df['atr'] = g_code['tr'].transform(lambda x: x.rolling(20).mean())
    df['ma20'] = g_code['close'].transform(lambda x: x.rolling(20).mean())
    df['ma200'] = g_code['close'].transform(lambda x: x.rolling(200).mean())
    df['bias'] = (df['close'] - df['ma20']) / df['ma20']
    
    # 活跃度 Z-Score
    v_mean = g_code['amount'].transform(lambda x: x.rolling(20).mean())
    v_std = g_code['amount'].transform(lambda x: x.rolling(20).std())
    df['f_active'] = (df['amount'] - v_mean) / (v_std + 1e-9)

    # 🚀 修正点 1：每日截面排序 (只跟当天的比)
    g_date = df.groupby('date')
    df['final_score'] = g_date['f_mom'].rank(pct=True) * 0.5 + \
                        g_date['f_active'].rank(pct=True) * 0.5
    
    # 偏离度惩罚
    df.loc[df['bias'] > 0.15, 'final_score'] = 0
    
    # 🚀 修正点 2：涨跌停判定
    df['is_limit_up'] = df['open'] > df['close'].shift(1) * 1.095
    # 修正这里的条件，去掉乘负号，改为跌幅超过9.5%
    df['is_limit_down_flat'] = (df['high'] == df['low']) & (df['close'] < df['close'].shift(1) * 0.905)
    
    return df

# ================= 3. 钢铁实战引擎 =================
class ProV25Engine:
    def __init__(self, df):
        self.df = df[df['date'] >= pd.to_datetime(CONFIG['backtest_start'])]
        self.capital = CONFIG['initial_capital']
        self.positions = {}
        self.history = []
        self.trades_log = []

    def run(self):
        dates = sorted(self.df['date'].unique())
        data_by_date = {d: day.set_index('code') for d, day in self.df.groupby('date')}
        
        for date in dates:
            day_df = data_by_date[date]
            
            # A. 卖出管理 (含跌停卖不出过滤)
            to_sell = []
            for code, pos in self.positions.items():
                if code not in day_df.index: continue
                r = day_df.loc[code]
                
                # 更新止损：最高价回落 2.0 倍 ATR
                pos['stop_loss'] = max(pos['stop_loss'], r['high'] - 2.0 * r['atr'])
                
                if r['low'] < pos['stop_loss']:
                    # 🚀 实战修正：一字跌停卖不出
                    if r['is_limit_down_flat']: 
                        continue 
                    
                    sell_p = pos['stop_loss'] * (1 - CONFIG['slippage'])
                    val = pos['shares'] * sell_p * (1 - CONFIG['fee'])
                    self.capital += val
                    self.trades_log.append({
                        '代码': code, '买入日期': pos['date'], '卖出日期': date.strftime('%Y-%m-%d'),
                        '盈利金额': round(val - pos['cost'], 2), '盈亏%': round((sell_p/pos['entry_p']-1)*100, 2)
                    })
                    to_sell.append(code)
            for c in to_sell: del self.positions[c]
            
            # B. 买入管理 (含涨停买不进过滤)
            if len(self.positions) < CONFIG['max_positions']:
                # 信号：200日线上 且 分数排在前面
                cands = day_df[(day_df['final_score'] > 0) & (day_df['close'] > day_df['ma200'])]
                cands = cands.sort_values('final_score', ascending=False).head(15)
                
                for code, r in cands.iterrows():
                    if len(self.positions) >= CONFIG['max_positions']: break
                    if code in self.positions: continue
                    # 🚀 实战修正：开盘涨停买不到
                    if r['is_limit_up']: continue
                    
                    risk_amt = self.capital * CONFIG['risk_per_trade']
                    stop_dist = 2.0 * r['atr'] if r['atr'] > 0 else r['close'] * 0.08
                    shares = (risk_amt / stop_dist // 100) * 100
                    
                    # 容量限制
                    max_sh = (r['amount'] * CONFIG['capacity_limit'] / r['open'] // 100) * 100
                    final_sh = min(shares, max_sh)
                    
                    buy_p = r['open'] * (1 + CONFIG['slippage'])
                    cost = final_sh * buy_p * (1 + CONFIG['fee'])
                    
                    if final_sh >= 100 and self.capital >= cost:
                        self.capital -= cost
                        self.positions[code] = {
                            'shares': final_sh, 'entry_p': buy_p, 'stop_loss': buy_p - stop_dist,
                            'date': date.strftime('%Y-%m-%d'), 'atr': r['atr'], 'cost': cost
                        }

            cur_v = self.capital + sum(p['shares'] * (day_df.loc[c,'close'] if c in day_df.index else p['entry_p']) for c,p in self.positions.items())
            self.history.append({'date': date, 'equity': cur_v})
            
        return pd.DataFrame(self.history).set_index('date'), pd.DataFrame(self.trades_log)

# ================= 4. 执行 =================
if __name__ == "__main__":
    conn = sqlite3.connect(CONFIG['db_path'])
    df = pd.read_sql("SELECT * FROM daily_data WHERE date >= '2023-01-01'", conn)
    conn.close()
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['code', 'date'])
    df = build_factor_score(df)
    
    engine = ProV25Engine(df)
    equity, trades = engine.run()
    
    csv_p = f"{CONFIG['out_dir']}/Zeus_Pro_V25_Trades.csv"
    trades.to_csv(csv_p, index=False, encoding='utf_8_sig')
    
    total_ret = (equity['equity'].iloc[-1] / CONFIG['initial_capital'] - 1) * 100
    max_dd = ((equity['equity'] - equity['equity'].cummax()) / equity['equity'].cummax()).min() * 100
    
    print(f"\n" + "🛡️"*15)
    print(f"🏆 Zeus Pro V2.5 (实战钢铁修复版)")
    print(f"📈 最终净收益: {total_ret:.2f}%")
    print(f"📉 最大回撤: {max_dd:.2f}%")
    print(f"📂 报表位置: {csv_p}")
    print(f"🛡️"*15)
    
    plt.style.use('dark_background')
    equity['equity'].plot(color='#00FF00', figsize=(10, 6), title='Zeus Pro V2.5 Performance')
    plt.savefig(f"{CONFIG['out_dir']}/Zeus_Pro_V25_Chart.png")
