import numpy as np
import matplotlib.pyplot as plt

def load_data():
    records = [[1,1,-1], [1,2,-1], [1,3,-1], [2,1,-1], [2,2,-1], [2,3, +1], [3,1,-1], [3,2,+1], [3,3,+1]]  
    data = []
    label = []
    for record in records:
        #add one dimension-----x0
        data.append([1, float(record[0]), float(record[1])])
        label.append(int(record[2]))
    return data, label

def sign(weight, x):
    if weight.dot(x) < 0:
        return -1
    elif weight.T.dot(x) == 0:
        return 0
    else:
        return +1

def pocket(data, label):
    data = np.array(data); label = np.array(label)
    # m = 2 + 1
    m, n = data.shape
    #weight before revising
    weight_old = np.zeros((n))
    #weight after revising
    weight_new = np.zeros((n))
    #best w
    weight_best = np.zeros((n))
    best_error_count = 0
    new_error_count = 0
    for indexs in range(1000):
        for i in range(m):
            best_error_count = 0
            new_error_count = 0
            predict = sign(weight_old, data[i])
            if predict != label[i]:
                all_changed = True
                weight_new = weight_old + label[i] * data[i]
            #everytime new weight is calculated, it compares with best weight in the whole dataset.less mistake? then reassign best weight
                for index in range(m):
                    best_prediction = sign(weight_best, data[index])
                    new_prediction = sign(weight_new, data[index])
                    if best_prediction != label[index]:
                        best_error_count += 1
                    if new_prediction != label[index]:
                        new_error_count += 1
                if new_error_count <= best_error_count:
                    weight_best = weight_new
                weight_old = weight_new
    return weight_best.tolist()  

def visualize(data, label, weight):
    xcord_1 = []
    ycord_1 = []
    xcord1 = [] 
    ycord1 = []
    for i in range(len(label)):
        if label[i] == -1:
            xcord_1.append(data[i][1])
            ycord_1.append(data[i][2])
        else:
            xcord1.append(data[i][1])
            ycord1.append(data[i][2])
    plt.switch_backend("agg")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #draw a straight line
    x0 = np.arange(-4, 6, 0.1)
    y0 = (-weight[0] - weight[1] * x0) / weight[2]
    plt.ylim(-4, 6)
    ax.plot(x0, y0, color = "black")
    
    ax.scatter(xcord_1, ycord_1, s = 30, c = "red", marker = "o", alpha = 1, label = "-1")
    ax.scatter(xcord1, ycord1, s = 30, c = "blue", marker = "+", alpha = 1,  label = "+1")

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    plt.legend()


    # dpi = 300 means 1800 * 1200 size of a graph
    plt.rcParams['savefig.dpi'] = 300 
    plt.savefig("visualize")
    plt.close()
   
if __name__ == "__main__":
    data, label = load_data()
    weight = pocket(data, label)
    visualize(data, label, weight)
