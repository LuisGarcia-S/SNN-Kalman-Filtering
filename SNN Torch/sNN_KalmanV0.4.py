import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from tqdm import tqdm

import torch
import torch.nn as nn
import snntorch as snn
import snntorch.spikegen as spkgen


sqrtpi = np.sqrt(np.pi)
a = 0.565
class StepFowardEncoder:
    def __init__(self, Ic,  thr, n_signals):
        self.base = torch.zeros(2, n_signals)
        self.thr = thr
        self.Ic = Ic
    
    def encode(self, x):
        self.base[0] = self.base[1]
        self.base[1] = torch.Tensor(x)
        spk_out = spkgen.delta(self.base, padding=True, off_spike=True, threshold=self.thr)

        return torch.where(spk_out[1]>0, 1, 0)*self.Ic, torch.where(spk_out[1]<0, 1, 0)*self.Ic,


class SFDecoderModified:
    def __init__(self, thr):
        self.base = 0
        self.thr = thr
        self.out_hist = []

    def decode(self,spk_pos, spk_neg):
        self.base = self.base + self.thr * spk_pos - self.thr * spk_neg
        self.out_hist.append(self.base)
        return self.base


class Network:
    def __init__(self, n, m, **kwargs):
        self.n = n
        self.m = m
        self.alpha = 1          #learn ratio

        self.wmin = -1.0e-3            #peso sinaptico minimo 1mS (Conductancia)
        #self.wmin = 1            #peso sinaptico minimo 1mS (Conductancia)
        self.wmax = 1.0e-3        #peso sinaptico max 1000mS (Conductancia)

        default_params = {
                'R':10e6,
                'C':1e-9,
                'thr':15e-3,
                'dt':0.1e-3,
                'rst':'zero',
                'beta':0
                }

        params = {**default_params, **kwargs}

        self.weights = torch.rand(n,m) * 2*self.wmax + self.wmin
        self.mem_LIF = torch.zeros(1,n)
        self.mem_LIFC = torch.zeros(1,m)

        self.spk_LIF = torch.zeros(1,n)

        self.spk_LIFC = torch.zeros(1,m)
        self.current_LIFC = torch.zeros(1,m)


        if params['beta'] == 0:
            self.LIF = snn.Lapicque(R=params['R'],
                                    C=params['C'],
                                    threshold=params['thr'],
                                    time_step=params['dt'])
        else:
            self.LIF = snn.Lapicque(beta=params['beta'],
                                    threshold=params['thr'],
                                    time_step=params['dt'])

        self.LIFC = snn.Synaptic(alpha=self.LIF.beta,
                                            beta=self.LIF.beta,
                                            threshold=params['thr'])

    def simulate(self, Ic, error):
        Ic[-1] = Ic[-1]+ float(error*1e-9)
        self.spk_LIF, self.mem_LIF = self.LIF(Ic, self.mem_LIF)
        self.spk_LIFC, self.current_LIFC, self.mem_LIFC = self.LIFC(torch.mm(self.spk_LIF,self.weights), self.current_LIFC, self.mem_LIFC)

        self.learning_RSTDP()
        return self.spk_LIFC

    def learning_RSTDP(self, R = 1):
        delta_w = torch.zeros_like(self.weights) + self.spk_LIF.T
        delta_w = delta_w - self.spk_LIFC
        delta_w = delta_w * 0.01e-3 * 0.02
        self.weights = self.weights + delta_w
        torch.clamp(self.weights, self.wmin, self.wmax, out=self.weights)

dt = 0.1e-3

t_ini = 0
t_fin = 30 

tiempo = np.arange(t_ini, t_fin, dt)


x1, __ = sp.symbols('x_1 \delta_t')                 #desarrollando series de taylor del atractor de lorenz (5 iteraciones)
Ax = sp.Matrix([[-10, 10, 0],
                [28, -1, -x1],
                [0, x1, -8/3]])

I = sp.eye(3)                                     
taylor = (I + (Ax * dt)/1
           + (((Ax * dt)**2)/2)
           + (((Ax * dt)**3)/6)
           +  (((Ax * dt)**4)/24)
           +  (((Ax * dt)**5)/120))

A_nonlinear = sp.lambdify(x1, Ax)
A_taylor = sp.lambdify(x1, taylor)                      #aproximacion del sistema, con entradas de control cero
C = sp.Matrix([[1,0,0]])
sig_x   =  0.535612920393419                            #definiendo desviacion estandar de x
sig_y   =  0.25375994265013957                          #definiendo desviacion estandar de y
sig_z   =  0.5271084217292544                           #definiendo desviacion estandar de x
sig_x_ob =  0.3301155938699889                          #definiendo desviacion estandar de x

SNNP = Network(4,3, dt=dt)
SNNN = Network(4,3, dt=dt)

Encoders = StepFowardEncoder(1.5e-9, 0.00001, 4)
Decoders = [SFDecoderModified(0.00001) for __ in range(3)]

xt = sp.Matrix([[5],[20],[-5]])
hxtt = sp.Matrix([[1],[1],[1]])
Kg = sp.zeros(3,1)

x1_hist = []
x2_hist = []
x3_hist = []

hx1_hist = []
hx2_hist = []
hx3_hist = []

hist_weightsP = []
hist_weightsN = []

time_spk_LIFCP = []
time_spk_LIFCN = []

current_LIFCN = []
current_LIFCP = []
        
time_spk_LIFP = []
time_spk_LIFN = []

mid_time = 3
flag_mid = True

for t in tqdm(tiempo):
    if t > mid_time and flag_mid:
        sig_x = np.random.rand()*1#1.2                      #definiendo desviacion estandar de x
        sig_y = np.random.rand()*1#1.4                      #definiendo desviacion estandar de y
        sig_z = np.random.rand()*1#1.1                      #definiendo desviacion estandar de x
        sig_x_ob = np.random.rand()*1.0                     #definiendo desviacion estandar de x
        flag_mid = False


    wt = sp.Matrix([[np.random.normal(0, sig_x)],           #creando ruido para las senales de la planta 
                    [np.random.normal(0, sig_y)],           #este ruido no estará disponile para la red
                    [np.random.normal(0, sig_z)]])

    vt = sp.Matrix([[np.random.normal(0, sig_x_ob)]])   #creando ruido para las senales de la planta a la salida

    dxt = A_nonlinear(xt[0,0]) * xt + wt                #simulando la planta no lineal  y agregando ruido
    xt = xt + dt * dxt                                  #integrando euler
    yt = C * xt + vt                                    #obteniendo mediciones de la planta y agregando ruido de medicion   
                                   
                                                        #paso 1 prediccion del kalman filter
    hxtt1 = A_taylor(hxtt[0,0]) * hxtt                  #obteniendo estado predicho hxtt1
    hytt1 = C * hxtt1                                   #obteniendo salida estimada
    dhxt = hxtt - hxtt1                                 #calculando la diferencia entre lo predicho entre el paso anterior y la nueva prediccion
    dyt = yt - hytt1                                    #calculando la diferencia entre lo medido  y la nueva medicion predicha

    if abs(dyt[0,0]) > 100:
        break

    vector_input = [dhxt[0,0], dhxt[1,0], dhxt[2,0], dyt[0,0]]      #poniendo las senales de entrada de la red en un solo vector
    
    IcP, IcN = Encoders.encode(vector_input)                        #codificando

    dd_SNNP = SNNP.simulate(IcP, max(dyt[0,0], 0))                          #suministrando corriente de exitacion a la primer capa de la red SNNP, simulando salida 
    dd_SNNN = SNNN.simulate(IcN, min(dyt[0,0], 0))                          #suministrando corriente de exitacion a la primer capa de la red SNNN, simulando salida 

    for k, decoder in enumerate(Decoders):                          #iterando entre cada decoder para obtener la matriz de kalman
        Kg[k, 0] = decoder.decode(dd_SNNP[0,k], dd_SNNN[0,k])       #obteniendo nuevo componente de la matriz de kalman

    hxtt = hxtt1 + Kg * dyt                                         #obteniendo estimacion para el siguiente paso
    x1_hist.append(xt[0,0])
    x2_hist.append(xt[1,0])
    x3_hist.append(xt[2,0])
    
    hx1_hist.append(hxtt[0,0])
    hx2_hist.append(hxtt[1,0])
    hx3_hist.append(hxtt[2,0])

    hist_weightsP.append(torch.flatten(SNNP.weights))
    hist_weightsN.append(torch.flatten(SNNN.weights))

    time_spk_LIFP.append( torch.flatten(SNNP.spk_LIF)*t )
    time_spk_LIFN.append( torch.flatten(SNNN.spk_LIF)*t )

    time_spk_LIFCP.append(torch.flatten(SNNP.spk_LIFC)*t)
    time_spk_LIFCN.append(torch.flatten(SNNN.spk_LIFC)*t)

    current_LIFCP.append(torch.flatten(SNNP.current_LIFC))
    current_LIFCN.append(torch.flatten(SNNN.current_LIFC))
   
np.savetxt('state1.csv',
        x1_hist,
        delimiter=', ',
        fmt='% s')
np.savetxt('state2.csv',
        x2_hist,
        delimiter=', ',
        fmt='% s')
np.savetxt('state3.csv',
        x3_hist,
        delimiter=', ',
        fmt='% s')

np.savetxt('estimate_state1.csv',
        hx1_hist,
        delimiter=', ',
        fmt='% s')
np.savetxt('estimate_state2.csv',
        hx2_hist,
        delimiter=', ',
        fmt='% s')
np.savetxt('estimate_state3.csv',
        hx3_hist,
        delimiter=', ',
        fmt='% s')

np.savetxt('hist_weighP.csv',
        hist_weightsP,
        delimiter=', ',
        fmt='% s')
np.savetxt('hist_weighN.csv',
        hist_weightsN,
        delimiter=', ',
        fmt='% s')

np.savetxt('spks_LIFP.csv',
        time_spk_LIFP,
        delimiter=', ',
        fmt='% s')
np.savetxt('spks_LIFN.csv',
        time_spk_LIFN,
        delimiter=', ',
        fmt='% s')

np.savetxt('spks_LIFCP.csv',
        time_spk_LIFCP,
        delimiter=', ',
        fmt='% s')
np.savetxt('spks_LIFCN.csv',
        time_spk_LIFCN,
        delimiter=', ',
        fmt='% s')


np.savetxt('current_LIFCP.csv',
        current_LIFCP,
        delimiter=', ',
        fmt='% s')
np.savetxt('current_LIFCN.csv',
        current_LIFCN,
        delimiter=', ',
        fmt='% s')

fig, ax = plt.subplots(3)
#ax = plt.axes(projection='3d')
ax[0].plot(tiempo[:len(x1_hist):9], x1_hist[::9],label='$x1$')
ax[1].plot(tiempo[:len(x2_hist):9], x2_hist[::9],label='$x2$')
ax[2].plot(tiempo[:len(x3_hist):9], x3_hist[::9],label='$x3$')
ax[0].plot(tiempo[:len(hx1_hist):9], hx1_hist[::9], label='$\hat x1$')
ax[1].plot(tiempo[:len(hx2_hist):9], hx2_hist[::9], label='$\hat x2$')
ax[2].plot(tiempo[:len(hx3_hist):9], hx3_hist[::9], label='$\hat x3$')

ax[0].legend()
ax[1].legend()
ax[2].legend()

plt.figure(num='Error')
plt.plot(tiempo[:len(x1_hist):9], np.array(x1_hist)[::9]-np.array(hx1_hist)[::9])
plt.plot(tiempo[:len(x2_hist):9], np.array(x2_hist)[::9]-np.array(hx2_hist)[::9])
plt.plot(tiempo[:len(x3_hist):9], np.array(x3_hist)[::9]-np.array(hx3_hist)[::9])

plt.figure(num='Pesos Red Positiva')
plt.plot(tiempo[:len(x1_hist):9],torch.stack(hist_weightsP)[::9])

plt.figure(num='Pesos Red Negativa')
plt.plot(tiempo[:len(x1_hist):9],torch.stack(hist_weightsN)[::9])

plt.figure(num='Kalman gain')
for data in Decoders:
    plt.plot(tiempo[:len(hx1_hist):5], data.out_hist[::5])

plt.show()

