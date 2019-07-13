import matplotlib.pyplot as plt
import os

def epoch_accuracy(results, outfile):
    print("plotting epoch-accuracy")
    plt.clf()

    legend = []
    colors = 'bgrcmy'
    styles = {"train acc":'','test acc':'--'}
    max_x = 0
    max_y = 0
    for modelname, color in zip(results, colors):
        acc = results[modelname]
        for is_train, style in styles.items():
            x = sorted(acc[is_train])
            y = [acc[is_train][_x] for _x in x]
            max_x = max(max_x,max(x))
            max_y = max(max_y,max(y))
            style = styles[is_train]
            plt.plot(x, y, color + style)
            legend.append(modelname+" "+is_train[:-4])

    max_y = min(max_y+.05,1)

    plt.gca().legend(legend)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.axis([0,max_x, max_y-0.25, max_y])
    plt.savefig(outfile)
    plt.show()


def numE_accuracy(results, outfile, predesigned_result=None, z_size=None):
    print("plotting numE-accuracy")
    plt.clf()

    legend = []
    colors = 'bgrcmy'
    """
    max_x = 0
    max_y = 0
    """

    end = max([acc for numE in results for acc in results[numE]["train acc"]])

    results = {"train end": {numE:results[numE]["train acc"][end] for numE in results},
           "test end": {numE:results[numE]["test acc"][end] for numE in results},
           "test max": {numE:max(results[numE]["test acc"].values()) for numE in results}}
    if predesigned_result is not None:
        predesigned_result = {"train end": predesigned_result["train acc"][end],
                              "test end": predesigned_result["test acc"][end],
                              "test max": max(predesigned_result["test acc"].values())}


    """
    all_y = [a for d in results.values() for a in d]
    max_y = max(all_y)
    min_y = min(all_y)
    max_y = min(max_y + .05, 1)
    min_y = max(min_y - .05, 0)
    """
    keys = list(results.keys())
    for key, color in zip(keys, colors):
        result = results[key]
        x = sorted(result)
        y = [result[_x] for _x in x]
        plt.plot(x, y, color)
        legend.append(key)
    if predesigned_result is not None:
        for key, color in zip(keys, colors):
            x = z_size
            y = predesigned_result[key]
            plt.plot(x,y,color,marker="o")
            legend.append("predesigned "+key)

    #max_y = min(max_y+.05,1)

    plt.gca().legend(legend)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    #plt.axis([0,max_x, max_y-0.25, max_y])
    plt.savefig(outfile)
    plt.show()

def y_activation(results, case, experti, folder, groupsize=None):
    plt.clf()
    result = results[case]["y activation"][:,experti]
    classes = [str(i) for i in range(len(result))]
    if groupsize is None:
        plt.bar(classes, result)
    else:
        from itertools import islice, cycle
        color = ["blue"]*groupsize + ["green"]*groupsize
        color = list(islice(cycle(color),len(result)))
        plt.bar(classes, result, color=color)

    plt.xlabel("class")
    plt.ylabel("activation")
    plt.savefig(os.path.join(folder,"%s-%i y activation.png"%(case,experti)))
    plt.show()


def plot_epoch_accuracy(acc, outfile):
    print("plotting accuracy")
    plt.clf()

    legend = []
    colors = 'bgrcmy'
    styles = {"train":'','test':'--'}
    max_x = 0
    max_y = 0
    for modelname, color in zip(acc, colors):
        modelacc = acc[modelname]
        for is_train, acc_history in modelacc.items():
            x = range(len(acc_history))
            max_x = max(max_x,max(x))
            y = acc_history
            max_y = max(max_y,max(y))
            style = styles[is_train]
            plt.plot(x, y, color + style)
            legend.append(modelname+" "+is_train)

    max_y = min(max_y+.05,1)

    plt.gca().legend(legend)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.axis([0,max_x, max_y-0.25, max_y])
    plt.savefig(outfile)
    plt.show()

def plot_numE_accuracy(acc, outfile):
    print("plotting accuracy")
    plt.clf()

    legend = []
    colors = 'bgrcmy'

    max_x = max(acc)
    num_e = sorted(acc)
    acc = {"train end":[acc[numE]["train"][-1] for numE in sorted(acc)],
           "test end":[acc[numE]["test"][-1] for numE in sorted(acc)],
           "test max":[max(acc[numE]["test"]) for numE in sorted(acc)]}

    all_y = [a for d in acc.values() for a in d]
    max_y = max(all_y)
    min_y = min(all_y)
    max_y = min(max_y+.05,1)
    min_y = max(min_y-.05,0)
    for (key, history), color in zip(acc.items(),colors):
        plt.plot(num_e, history, color)
        legend.append(key)
    plt.gca().legend(legend)
    plt.xlabel('number of expert')
    plt.ylabel('accuracy')
    plt.axis([0, max_x, min_y, max_y])
    plt.savefig(outfile)
    plt.show()

def special_num_E_accuracy(acc, acc_special, special_x, outfile):
    print("plotting accuracy")
    plt.clf()

    legend = []
    colors = 'bgrcmy'


    max_x = max(acc)
    max_x = max(max_x,special_x)
    num_e = sorted(acc)
    acc = {"train end":[acc[numE]["train"][-1] for numE in sorted(acc)],
           "test end":[acc[numE]["test"][-1] for numE in sorted(acc)],
           "test max":[max(acc[numE]["test"]) for numE in sorted(acc)]}

    acc_special= {"train end":acc_special["train"][-1],
                  "test end":acc_special["test"][-1],
                  "test max":max(acc_special["test"])}

    all_y = [a for d in acc.values() for a in d]
    max_y = max(all_y)
    max_y = max(max_y, max(acc_special.values()))
    min_y = min(all_y)
    max_y = min(max_y+.05,1)
    min_y = max(min_y-.05,0)
    for (key, history), color in zip(acc.items(),colors):
        plt.plot(num_e, history, color)
        legend.append(key)
    for (key, y), color in zip(acc_special.items(), colors):
        plt.plot(special_x, y, color, marker="o")
        legend.append("special "+key)

    plt.gca().legend(legend)
    plt.xlabel('number of expert')
    plt.ylabel('accuracy')
    plt.axis([0, max_x, min_y, max_y])
    plt.savefig(outfile)
    plt.show()


def plot_activation_of_expert(activation_per_class_per_expert, expert_id, outfile):
    print("plot activation per class")
    plt.clf()

    activation_per_class = activation_per_class_per_expert[:,expert_id]
    classes = [str(i) for i in range(len(activation_per_class))]
    plt.bar(classes, activation_per_class)
    plt.xlabel("class")
    plt.ylabel("activation")
    plt.savefig(outfile)
    plt.show()

"""
def plot(outfile, xys, xlabel=None, ylabel=None, **kwargs):
    print("plot")
    plt.clf()
    color = "bgrcmy"
    def ks(i):
        return list({key[i] for key in xys})
    
        
    def style(i,k):
        if i == 0:
            
    
    for key in xys:
        x,y = xys[key]
        style = "".join([ for i,k in enumerate(key)])
        plt.plot(x,y,)
    
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.xlabel(ylabel)

    plt.savefig(outfile)
    plt.clf()
"""
