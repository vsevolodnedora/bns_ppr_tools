
import numpy as np

class FORMULAS:

    @staticmethod
    def vinf(eninf):
        return np.sqrt(2. * eninf)

    @staticmethod
    def vinf_bern(eninf, enthalpy):
        return np.sqrt(2.*(enthalpy*(eninf + 1.) - 1.))

    @staticmethod
    def vel(w_lorentz):
        return np.sqrt(1. - 1. / (w_lorentz**2))

    @staticmethod
    def get_tau(rho, vel, radius, lrho_b):

        rho_b = 10 ** lrho_b
        tau_0 = 0.5 * 2.71828182845904523536 * (radius / vel) * (0.004925794970773136) # in ms
        tau_b = tau_0 * ((rho/rho_b) ** (1.0 / 3.0))
        return tau_b # ms

    @staticmethod
    def enthalpy(eps, press, rho):
        return 1 + eps + (press / rho)
