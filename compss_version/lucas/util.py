import numpy as np
np.set_printoptions(threshold=np.nan)


def plot_histogram(pred,name):
    import matplotlib.pyplot as plt
    plt.clf()
    plt.hist(pred, bins=5)
    plt.title("Predictions Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.xlim((0,1))
    #plt.xticks([0,.20,.40,.5,.60,.7,0.8,0.9,1])
    plt.savefig(name)


def plot_bars(pred,name):
    import matplotlib.pyplot as plt
    plt.clf()
    plt.hist(pred, bins=2)
    plt.title("Predictions Frequency")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.xlim((0,1))
    plt.xticks([0,1])
    plt.savefig(name)


def plot_ROC(actual, predictions):
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    plt.clf()
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b',
    label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    #plt.show()
    plt.savefig('ROC.png')



def plot_precision_recall(actual,predictions):
    from sklearn.metrics import precision_recall_curve
    precision, recall, threshold = precision_recall_curve(actual,predictions,pos_label=1)
    import matplotlib.pyplot as plt

    plt.clf()
    plt.plot(recall, precision)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall')
    plt.legend(loc="upper right")
    plt.savefig('Precision-Recall.png')



# Enviroment to SET
SAVE_TABLE_TO_PLOT = False
ValidGridsFile = '/home/lucasmsp/workspace/BigSea/waze-jams/compss_version/lucas/ValidGrids.csv'
forecastsFile  = "/home/lucasmsp/workspace/BigSea/waze-jams/compss_version/lucas/forecasts_curitiba.csv"
oficialFile    = '/home/lucasmsp/workspace/BigSea/waze-jams/compss_version/lucas/result.txt'

validGrids    = np.loadtxt(ValidGridsFile, delimiter=',',dtype=int)
ngrids = len(validGrids)
print "Number of Grids {}".format(ngrids )
pred = []
if SAVE_TABLE_TO_PLOT:
    f_out = open("table_colors.csv",'w')
    grid = 0
for line in open(forecastsFile, 'r'):
    lines = line.split(",")
    try:
        if int(lines[0]) in validGrids:
            pred.append([float(lines[2]),float(lines[3])])
            if SAVE_TABLE_TO_PLOT:
                f_out.write('{},{}\n'.format(grid,lines[2]))
                grid+=1
    except Exception as e:
        pass

pred = np.array(pred)

#pred = np.loadtxt('/home/lucasmsp/workspace/BigSea/waze-jams/compss_version/lucas/table_colors.csv', delimiter=',', dtype=float)
true = np.loadtxt(oficialFile,       delimiter=',', dtype=int)
true = true[validGrids]
print len(true)
acc=0
for pv,t in zip(pred,true):
    p,v = pv
    pr = int(np.rint(p))
    #print "{} ({}) ~ {}".format(pr,p,t)
    if pr == t:
        acc+=1
acc = float(acc)/ngrids
print "Accuracy: {}".format(acc)


plot_bars(true,'Plot_RealData.png')
plot_bars(np.rint(pred[:,0]),'Plot_PredictionRounded.png')
plot_histogram(pred[:,0],'Histogram_PredictionData.png')
#plot_ROC(true, pred[:,0])
#plot_precision_recall(true, pred[:,0])
