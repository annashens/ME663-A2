import numpy as np
import pandas as pd

class Physics:
    def __init__(self, scheme, gamma=1.4):
        self.gamma = gamma
        self.scheme = scheme

    def U_to_primitive(self, U):
        rho = U[0]
        u   = U[1]/rho
        E   = U[2]/rho
        p   = (self.gamma-1)*(rho*E - 0.5*rho*u**2)
        a   = np.sqrt(self.gamma*p/rho)
        return rho, u, E, p, a

    def lp(self, u, a, rho):
        if self.scheme == 'sw':
            return [max(u, 0.0), max(u + a, 0.0), max(u - a, 0.0)]
        elif self.scheme == 'vl':
            M = u/a
            if M <= -1.0:
                return [0.0, 0.0, 0.0]
            if M >= 1.0:
                return [u - a, u, u + a]
            gamma = self.gamma
            
            l1 = (1/4)*a*(M + 1)**2*(-(M - 1)**2/(gamma + 1) + 1)
            l2 = (1/4)*a*(M + 1)**2*(-M + (M - 1)**2*(gamma - 1)/(gamma + 1) + 3)
            l3 = (1/2)*a*(M - 1)*(M + 1)**2*(M*((1/2)*gamma - 1/2) + 1)/(gamma + 1)

            return [l1, l2, l3]
        else:
            raise ValueError('Unknown flux scheme')

    def lm(self, u, a, rho):
        if self.scheme == 'sw':
            return [min(u, 0.0), min(u + a, 0.0), min(u - a, 0.0)]
        elif self.scheme=='vl':
            M = u/a 
            if M >= 1.0:
                return [0.0, 0.0, 0.0]
            if M <= -1.0:
                return [u - a, u, u + a]
            gamma = self.gamma
            
            fm1 = -1/4*a*rho*(M - 1)**2
            l1 = -1/4*a*(1 - M)**2*(-(-M - 1)**2/(gamma + 1) + 1)
            l2 = -1/2*a*(1 - M)**2*(-M - 1)*(-M*((1/2)*gamma - 1/2) + 1)/(gamma + 1)
            l3 = -1/4*a*(1 - M)**2*(M + (-M - 1)**2*(gamma - 1)/(gamma + 1) + 3)
            return [l1, l2,l3 ]
        else:
            raise ValueError('Unknown flux scheme')

    def build_F(self, rho, u, a, L):
        gamma = self.gamma
        l1, l2, l3 = L
        
        # Derived in Q3
        F1 = 2*(gamma-1)*l1 + l2 + l3
        F2 = 2*(gamma-1)*l1*u + l2*(u+a) + l3*(u-a)
        F3 = ((gamma-1)*l1*u**2
              + l2/2*(u+a)**2
              + l3/2*(u-a)**2
              + (3-gamma)/(2*(gamma-1)) * (l2+l3)*a**2
              )

        return (rho/(2*gamma)) * np.array([F1, F2, F3])

class Solver:
    def __init__(self, scheme, xmin, xmax, N, wl, wr, gamma=1.4, CFL=0.4, Umax=None):
        self.phys = Physics(gamma=gamma, scheme=scheme)
        self.flux = scheme
        self.CFL = CFL
        self.Umax = Umax
        
        self._build_grid(xmin, xmax, N) # build grid
        self._build_IC(wl, wr) # apply initial conditions

        self.F = np.zeros((self.N,3))

    def _build_grid(self, xmin, xmax, N):
        self.N = N+1
        dx = (xmax - xmin)/(N)
        self.dx = dx
        # cell-centered grid
        # self.x =xmin + 0.5*dx + np.arange(N)*dx
        self.x = np.linspace(xmin, xmax, N+1)
    def _build_IC(self, wl, wr):
        self.U = np.zeros((self.N,3))

        for i in range(self.N):
            w = wl if self.x[i] < 0 else wr 
            rho, u, p = (w[var] for var in ('rho', 'u', 'p'))

            E = p/((self.phys.gamma-1)*rho) + 0.5*u**2
            self.U[i] = [rho, rho*u, rho*E]
            
    def run(self, t_start, t_end, return_df = True):
        t = t_start
        step = 0

        while t < t_end:
            self.dt = self._calc_dt() # calculate time step based on CFL condition
            self.F = self._calc_flux()
            self.U = self._calc_u()
            t += self.dt
            step += 1
        return self.get_solution(return_df)
    def _calc_flux(self):
        F = np.zeros((self.N-1, 3))
        for i in range(self.N-1):
            for offset in [0,1]:
                rho, u, _, _, a = self.phys.U_to_primitive(self.U[i+offset])
                if offset == 0: # left state
                    Fp = self.phys.build_F(rho, u, a, self.phys.lp(u, a, rho))
                else: # right state
                    Fm = self.phys.build_F(rho, u, a, self.phys.lm(u, a, rho))
            F[i] = Fp + Fm
        return F

    def _calc_u(self):
        U_new = self.U.copy()

        for i in range(1, self.N-1):
            # euler update: U_new = U_old - (dt/dx)*(F[i] - F[i-1])
            U_new[i] = self.U[i] - (self.dt/self.dx)*(self.F[i] - self.F[i-1])
            
        # enforce zero-gradient (Neumann) boundary conditions
        U_new[0]  = U_new[1]
        U_new[-1] = U_new[-2]

        return U_new

    def _calc_dt(self):
        if self.Umax is not None:
            return self.dx*self.dx / self.Umax
        else:
            max_speed = 0
            for i in range(self.N):
                _, u, _, _, a = self.phys.U_to_primitive(self.U[i])
                max_speed = max(max_speed, abs(u) + a)
            # CFL condition: dt = CFL * dx / max_speed
            return self.CFL * self.dx / max_speed

    def get_solution(self, return_df = True):
        rho = np.zeros(self.N)
        u = np.zeros(self.N)
        E = np.zeros(self.N)
        p = np.zeros(self.N)
        a = np.zeros(self.N)

        for i in range(self.N):
            rho[i], u[i], E[i], p[i], a[i] = self.phys.U_to_primitive(self.U[i])

        if return_df:
            return pd.DataFrame({
                'x': self.x,
                'density': rho,
                'velocity': u,
                'total_energy': E,
                'pressure': p,
                'speed_of_sound': a,
                'mach': np.abs(u)/a
            })
        return rho, u, E, p, a
