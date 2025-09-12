from AlgorithmImports import *
import numpy as np
import math

RISK_FREE_ANNUAL = 0.05

class Momentum1(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2014, 1, 1)
        self.SetCash(1_000_000)
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)
        self.SetBenchmark("SPY")
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.CoarseSelection, self.FineSelection)

        self.lookbacks = [63, 252]
        self.targetVol = 0.1
        self.maxPos = 0.05
        self.rebalance = Resolution.Daily
        self.lastRebalance = None

        self.symbolData = {}
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        
        # Add VIX for regime filtering
        try:
            self.vix = self.AddData(CBOE, "VIX", Resolution.Daily).Symbol
        except:
            self.vix = None
            self.Debug("VIX data not available")

        self.slippage = 0.0005
        self.transactionFee = 0.0001

        self.Schedule.On(self.DateRules.Every(DayOfWeek.Monday), self.TimeRules.At(10, 0), self.Rebalance)

    def CoarseSelection(self, coarse):
        filtered = [c for c in coarse if c.HasFundamentalData and c.Price > 5.0]
        byDollarVol = sorted(filtered, key=lambda c: c.DollarVolume, reverse=True)[:500]
        return [c.Symbol for c in byDollarVol]

    def FineSelection(self, fine):
        return [f.Symbol for f in fine]

    def Rebalance(self):
        # Regime filter check
        regime_ok = True
        if self.vix and self.Securities[self.vix].Price > 0:
            regime_ok = VIX<25
            
        if not regime_ok:
            for kv in self.Portfolio:
                if kv.Value.Invested and kv.Key != self.spy:
                    self.Liquidate(kv.Key)
            return

        candidates = [s for s in self.ActiveSecurities.Keys if s != self.spy and s != self.vix]
        scores = []
        
        for s in candidates:
            try:
                hist = self.History(s, 260, Resolution.Daily)
                if hist.empty: 
                    continue
                px = hist['close'].values
                
                # Momentum calculation
                mom_scores = []
                for lb in self.lookbacks:
                    if len(px) > lb:
                        mom = px[-1]/px[-lb] - 1
                        mom_scores.append(mom)
                
                if mom_scores:
                    avg_momentum = np.nanmean(mom_scores)
                    scores.append((s, avg_momentum))
            except Exception as e:
                self.Debug(f"Error processing {s}: {e}")
                continue

        scores.sort(key=lambda x: x[1], reverse=True)
        top = [s for s, _ in scores[:20]]

        # Volatility targeting
        weights = {}
        for s in top:
            try:
                ret_hist = self.History(s, 63, Resolution.Daily)
                if ret_hist.empty:
                    continue
                    
                returns = ret_hist['close'].pct_change().dropna()
                if len(returns) < 20: 
                    continue
                    
                vol = float(returns.std() * math.sqrt(252))
                if vol == 0: 
                    continue
                    
                target_weight = min(self.maxPos, (self.targetVol/vol)/len(top))
                weights[s] = target_weight
            except Exception as e:
                self.Debug(f"Error calculating weight for {s}: {e}")
                continue

        # Execute trades
        targets = [(s, w) for s, w in weights.items()]
        self.SetHoldings(targets)

    def OnSecuritiesChanged(self, changes):
        for sec in changes.AddedSecurities:
            self.symbolData[sec.Symbol] = True

    def OnOrderEvent(self, orderEvent):
        pass
