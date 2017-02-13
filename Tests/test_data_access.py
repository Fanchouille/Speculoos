from DataManagement import DataHandler as datHand

# .PA means PARIS STOCK EXCHANGE
PARAMS = {'homepath': '/Users/fanch/Desktop/Titres', 'stocklist' : ['SLB.PA','AF.PA']}



handData = datHand.DataHandler(PARAMS)
#handData.save_all_stocks()

print handData.check_consistency('SLB.PA')
print handData.check_consistency('AF.PA')

