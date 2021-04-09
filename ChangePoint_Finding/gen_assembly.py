from matplotlib import rcParams
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial"]
rcParams.update({'font.size': 18})
import numpy as np
import matplotlib.pyplot as plt

def nordata(data):
    m = np.mean(data)
    s = np.std(data)
    return (data-m)/s

# def gen_assembly(BM_initial, sd_initial, t_initial, BM_final, sd_final, t_final ):
def gen_assembly(BM, t_change, noise=[5,5], type='growing'):
    if type == 'growing':
        BM_initial = min(BM)
        sd_initial = min(noise)
        BM_final = max(BM)
        sd_final = max(noise)
    else:
        BM_initial = max(BM)
        sd_initial = max(noise)
        BM_final = min(BM)
        sd_final = min(BM)
    t_initial = min(t_change)
    t_final = max(t_change)
    t_assembly = t_final - t_initial
    curve_initial = BM_initial * np.ones(t_initial) + np.random.normal(0, sd_initial, t_initial)
    curve_assembly = np.linspace(BM_initial, BM_final, t_assembly) + np.random.normal(0, (sd_initial+sd_final)/2, t_assembly)
    curve_final = BM_final * np.ones(t_final) + np.random.normal(0, sd_final, t_final)
    data_ori = np.append(np.append(curve_initial,curve_assembly), curve_final)
    t = np.linspace(1, t_initial+t_final+t_assembly, t_initial+t_final+t_assembly)
    return t, data_ori


## simluate data
#  setting parameters
BM_initial = 35
sd_initial = 5
t_initial = 150 ## first change point
BM_final = 10
sd_final = 3.5
t_final = 300 ## second change point


if __name__ == "__main__":
    t, data = gen_assembly(BM=[BM_initial, BM_final], t_change=[t_initial, t_final])
    data_nor = nordata(data)
    plt.plot(t, data)