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
uncovered_claims = data[0]['cumulative_uncovered_claims']
claims = data[0]['cumulative_claims']
insurance_firms_cash = data[0]['insurance_firms_cash']
uncovered_risks = data[0]['uncovered_risks']
#risk_models = data[0]['number_riskmodels']
syndicateA_cash = data[0]['syndicateA_cash']
syndicateB_cash = data[0]['syndicateB_cash']
syndicateC_cash = data[0]['syndicateC_cash']
syndicateD_cash = data[0]['syndicateD_cash']
syndicateE_cash = data[0]['syndicateE_cash']
syndicateF_cash = data[0]['syndicateF_cash']
syndicateA_contracts = data[0]['syndicateA_contracts']
syndicateB_contracts = data[0]['syndicateB_contracts']
syndicateC_contracts = data[0]['syndicateC_contracts']
syndicateD_contracts = data[0]['syndicateD_contracts']
syndicateE_contracts = data[0]['syndicateE_contracts']
syndicateF_contracts = data[0]['syndicateF_contracts']

rfile.close()

cs = contracts
pls = pl
os = op
hs = cash
ps = premium
ucl = uncovered_claims
uc = uncovered_risks

fig1 = plt.figure()
ax0 = fig1.add_subplot(711)
ax0.get_xaxis().set_visible(False)
ax0.plot(range(len(cs)), cs,"b")
ax0.set_ylabel("Contracts")
ax1 = fig1.add_subplot(712)
ax1.get_xaxis().set_visible(False)
ax1.plot(range(len(os)), os,"b")
ax1.set_ylabel("Active firms")
ax2 = fig1.add_subplot(713)
ax2.get_xaxis().set_visible(False)
ax2.plot(range(len(hs)), hs,"b")
ax2.set_ylabel("Cash")
ax3 = fig1.add_subplot(714)
ax3.get_xaxis().set_visible(False)
ax3.plot(range(len(pls)), pls,"b")
ax3.set_ylabel("Profits, Losses")
ax4 = fig1.add_subplot(715)
ax4.get_xaxis().set_visible(False)
ax4.plot(range(len(ps)), ps,"k")
ax4.set_ylabel("Premium")
ax5 = fig1.add_subplot(716)
ax5.get_xaxis().set_visible(False)
ax5.plot(range(len(ucl)), ucl,"k")
ax5.set_ylabel("Uncovered Claims")
ax6 = fig1.add_subplot(717)
ax6.plot(range(len(uc)), uc,"k")
ax6.set_ylabel("Uncovered Risks")
ax6.set_xlabel("Time")

plt.savefig("data/single_replication_systemic_information.pdf")

fig2 = plt.figure()
ax0 = fig2.add_subplot(611)
ax0.get_xaxis().set_visible(False)
ax0.plot(range(len(syndicateA_cash)), syndicateA_cash, "r")
ax0.set_ylabel("syndicateA cash")
ax1 = fig2.add_subplot(612)
ax1.get_xaxis().set_visible(False)
ax1.plot(range(len(syndicateB_cash)), syndicateB_cash, "r")
ax1.set_ylabel("syndicateB cash")
ax2 = fig2.add_subplot(613)
ax2.get_xaxis().set_visible(False)
ax2.plot(range(len(syndicateC_cash)), syndicateC_cash,"r")
ax2.set_ylabel("syndicateC cash")
ax3 = fig2.add_subplot(614)
ax3.get_xaxis().set_visible(False)
ax3.plot(range(len(syndicateD_cash)), syndicateD_cash, "r")
ax3.set_ylabel("syndicateD cash")
ax4 = fig2.add_subplot(615)
ax4.get_xaxis().set_visible(False)
ax4.plot(range(len(syndicateE_cash)), syndicateE_cash, "r")
ax4.set_ylabel("syndicateE cash")
ax5 = fig2.add_subplot(616)
ax5.get_xaxis().set_visible(False)
ax5.plot(range(len(syndicateF_cash)), syndicateF_cash,"r")
ax5.set_ylabel("syndicateF cash")
ax5.set_xlabel("Time")

plt.savefig("data/single_replication_individual_cash.pdf")



fig3 = plt.figure()
ax0 = fig3.add_subplot(611)
ax0.get_xaxis().set_visible(False)
ax0.plot(range(len(syndicateA_contracts)), syndicateA_contracts, "k")
ax0.set_ylabel("syndicateA contracts")
ax1 = fig3.add_subplot(612)
ax1.get_xaxis().set_visible(False)
ax1.plot(range(len(syndicateB_contracts)), syndicateB_contracts, "k")
ax1.set_ylabel("syndicateB contracts")
ax2 = fig3.add_subplot(613)
ax2.get_xaxis().set_visible(False)
ax2.plot(range(len(syndicateC_contracts)), syndicateC_contracts,"k")
ax2.set_ylabel("syndicateC contracts")
ax3 = fig3.add_subplot(614)
ax3.get_xaxis().set_visible(False)
ax3.plot(range(len(syndicateD_contracts)), syndicateD_contracts, "k")
ax3.set_ylabel("syndicateD contracts")
ax4 = fig3.add_subplot(615)
ax4.get_xaxis().set_visible(False)
ax4.plot(range(len(syndicateE_contracts)), syndicateE_contracts, "k")
ax4.set_ylabel("syndicateE contracts")
ax5 = fig3.add_subplot(616)
ax5.get_xaxis().set_visible(False)
ax5.plot(range(len(syndicateF_contracts)), syndicateF_contracts,"k")
ax5.set_ylabel("syndicateF cash")
ax5.set_xlabel("Time")

plt.savefig("data/single_replication_individual_contracts.pdf")

plt.show()


