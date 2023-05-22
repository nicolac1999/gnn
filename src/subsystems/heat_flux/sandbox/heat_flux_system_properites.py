from subsystems.heat_flux.utils.system_properties import delta_T_from_heat_flux, delta_T_kapitza_from_heat_power
import numpy as np
import matplotlib.pyplot as plt

# %%

cross_sections = np.array([30., 40])

heat_flux = np.arange(0., 60., 1.)

delta_T_30 = delta_T_from_heat_flux(heat_flux=heat_flux,
                                 cross_section=np.array([30.]),
                                    gradient_length=np.array([480.]),
                                    return_mK=True).flatten()

delta_T_60 = delta_T_from_heat_flux(heat_flux=heat_flux,
                                 cross_section=np.array([60.]),
                                    gradient_length=np.array([60.]),
                                    return_mK=True).flatten()

delta_T_150_480 = delta_T_from_heat_flux(heat_flux=heat_flux,
                                 cross_section=np.array([150.]),
                                    gradient_length=np.array([480.]),
                                     return_mK=True).flatten()

delta_T_150_270 = delta_T_from_heat_flux(heat_flux=heat_flux,
                                 cross_section=np.array([150.]),
                                    gradient_length=np.array([270.]),
                                     return_mK=True).flatten()

delta_Ts = [delta_T_30, delta_T_60, delta_T_150]
labels = ['cross section 30cm2', 'cross section 60cm2', 'cross section 150cm2']
plt.figure()
for i in range(len(delta_Ts)):
    plt.plot(heat_flux, delta_Ts[i], label=labels[i])
plt.xlabel('heat [ W ]')
plt.ylabel('delta T [ mK ]')
plt.yscale('log')
plt.title('ΔT for a given heat and a given cross section over 480 cm of length')
plt.legend()
plt.grid()
plt.show()


plt.figure()
plt.plot(heat_flux, delta_T_60)
plt.xlabel('heat [ W ]')
plt.ylabel('delta T [ mK ]')
plt.title('ΔT for a given heat and cross section of 60 cm2 over 60 cm of length')
#plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(heat_flux, delta_T_150_480)
plt.xlabel('heat [ W ]')
plt.ylabel('delta T [ mK ]')
plt.title('ΔT for a given heat and cross section of 150 cm2 over 480 cm of length')
#plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(heat_flux, delta_T_150_270)
plt.xlabel('heat [ W ]')
plt.ylabel('delta T [ mK ]')
plt.title('ΔT for a given heat and cross section of 150 cm2 over 270 cm of length')
#plt.legend()
plt.grid()
plt.show()

# %%
delta_T_kapitza_30 = delta_T_kapitza_from_heat_power(heat_power=heat_flux,
                                                     fraction_wetted_perim=np.array([0.3]))

delta_T_kapitza_20 = delta_T_kapitza_from_heat_power(heat_power=heat_flux,
                                                     fraction_wetted_perim=np.array([0.2]))
delta_T_kapitza_10 = delta_T_kapitza_from_heat_power(heat_power=heat_flux,
                                                     fraction_wetted_perim=np.array([0.1]))

delta_Ts_kapitza = [delta_T_kapitza_30, delta_T_kapitza_20, delta_T_kapitza_10]
labels = ['fraction 30 %', 'fraction 20 %', 'fraction 10 %']
plt.figure()
for i in range(len(delta_Ts_kapitza)):
    plt.plot(heat_flux, delta_Ts_kapitza[i], label=labels[i])
plt.xlabel('heat [ W ]')
plt.ylabel('delta T [ mK ]')
#plt.yscale('log')
plt.title('ΔT for a given heat and a given fraction of wetted perimeter for length 480 cm')
plt.legend()
plt.grid()
plt.show()



# %%

