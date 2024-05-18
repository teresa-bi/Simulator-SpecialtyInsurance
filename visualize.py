import matplotlib.pyplot as plt
import numpy as np

rfile = open("data/history_logs.dat","r")

data = [eval(k) for k in rfile]

cash = data[0]['total_cash']
excess_capital = data[0]['total_excess_capital']
pl = data[0]['total_profits_losses']
contracts = data[0]['total_contracts']
op = data[0]['total_operational']
premium = data[0]['market_premium']
bankruptcies = data[0]['cumulative_bankruptcies']
market_exits = data[0]['cumulative_market_exits']
unrecovered_claims = data[0]['cumulative_unrecovered_claims']
claims = data[0]['cumulative_claims']
insurance_firms_cash = data[0]['insurance_firms_cash']
#catastrophe_events = data[0]['rc_event_schedule_initial']
#catastrophe_events_damage = data[0]['rc_event_damage_initial']
#risk_models = data[0]['number_riskmodels']

rfile.close()

cs = contracts
pls = pl
os = op
hs = cash
ps = premium
ucl = unrecovered_claims

#cse = catastrophe_events
#csd = catastrophe_events_damage

fig1 = plt.figure()
ax0 = fig1.add_subplot(611)
ax0.get_xaxis().set_visible(False)
ax0.plot(range(len(cs)), cs,"b")
ax0.set_ylabel("Contracts")
ax1 = fig1.add_subplot(612)
ax1.get_xaxis().set_visible(False)
ax1.plot(range(len(os)), os,"b")
ax1.set_ylabel("Active firms")
ax2 = fig1.add_subplot(613)
ax2.get_xaxis().set_visible(False)
ax2.plot(range(len(hs)), hs,"b")
ax2.set_ylabel("Cash")
ax3 = fig1.add_subplot(614)
ax3.get_xaxis().set_visible(False)
ax3.plot(range(len(pls)), pls,"b")
ax3.set_ylabel("Profits, Losses")
ax4 = fig1.add_subplot(615)
ax4.get_xaxis().set_visible(False)
ax4.plot(range(len(ps)), ps,"k")
ax4.set_ylabel("Premium")
ax5 = fig1.add_subplot(616)
ax5.plot(range(len(ucl)), ucl,"k")
ax5.set_ylabel("Uncovered Claims")
ax5.set_xlabel("Time")

plt.savefig("data/single_replication_pt1.pdf")

plt.show()


