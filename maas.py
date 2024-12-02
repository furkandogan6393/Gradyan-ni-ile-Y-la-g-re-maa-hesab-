import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv(r"C:\Users\TozLu\Desktop\Dataset\Maas.csv")

veri = data.copy()
print(veri)

y_train = veri["Salary"]
x_train = veri["YearsExperience"]


def fonk(x,y,w,b):
    m=x_train.shape[0]
    total_hata=0
    for i in range(m):
        f_wb=(w * x[i]) + b
        total_hata += (f_wb-y[i])**2
    total_hata/=(2*m)
    return total_hata

def türev(x,y,w,b):
    m=x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = (w * x[i]) + b
        dj_dw += (f_wb - y[i]) * x[i]
        dj_db += (f_wb - y[i])
    dj_dw /= m
    dj_db /= m
    return dj_dw,dj_db

def wb_türev(x,y,w_in,b_in,num_iters, cost_türev, f_wb_fonk, alpha):
    w=w_in
    b=b_in
    
    for i in range(num_iters):
        dj_dw,dj_db=cost_türev(x,y,w,b)
        w -= (alpha*dj_dw)
        b -= (alpha*dj_db)
    return w,b

w_in=0
b_in=0
iterations=15000
alpha=0.01


w,b=wb_türev(x_train,y_train,w_in,b_in, iterations, türev, fonk, alpha)

m = x_train.shape[0]
predict = np.zeros(m)
for i in range(m):
    predict[i] = (w * x_train[i]) + b #Görüntüleme için
    

plt.scatter(x_train,y_train, marker="X", c="r", label="Gerçek Değerler")
plt.plot(x_train, predict, c="b", label="Tahmini değerler")
plt.title("YILA göre MAAŞ")
plt.xlabel("Çalıştığınız Yıl")
plt.ylabel("Almanız gereken Maaş")
plt.legend()
plt.show()

nüfus = float(input("Kaç yıldır çalıştığınızı girer misiniz: "))
predict1= (w * nüfus) + b
print(f"{nüfus}.Yıldır çalışan bir eleman için aylık ortalama Maaş: ${predict1/12:.2f}")
    