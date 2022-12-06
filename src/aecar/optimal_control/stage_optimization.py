
__all__ = ["StageOptimization"]

class StageOptimization:
    def __init__(self):

        def value_function(self,uc):
            cost = 0.0
            #x = np.copy(self.xc)
            x = np.zeros((self.n,5), dtype=complex)
            x[0,:] = self.x0
            for i in range(self.n-1):
                xspeed = (self.xdes[i+1,0] - self.xdes[i,0])/self.dt
                yspeed = (self.xdes[i+1,1] - self.xdes[i,1])/self.dt

                cost += (self.QP * (x[i,0]-self.xdes[i,0])**2 + self.QP * (x[i,1] - self.xdes[i,1])**2 +\
                    self.QS * (x[i,2]*np.cos(x[i,3])-xspeed)**2 + self.QS * (x[i,2] * np.sin(x[i,3]) - yspeed)**2 )
                cost += (self.R * (uc[i,0]**2 + uc[i,1]**2))

                # compute next state
                x[i+1,0] = x[i,0] + self.dt * x[i,2] * np.cos(x[i,3])
                x[i+1,1] = x[i,1] + self.dt * x[i,2] * np.sin(x[i,3])
                x[i+1,2] = x[i,2] + self.dt * uc[i,0] / self.m
                x[i+1,3] = x[i,3] + self.dt * x[i,2] * x[i,4] / self.L
                x[i+1,4] = x[i,4] + self.dt * uc[i,1] / self.I

            # add final state cost
            cost += self.QPF * (x[self.n-1,0]-self.xdes[self.n-1,0])**2 + self.QPF * (x[self.n-1,1] - self.xdes[self.n-1,1])**2
            return cost

    def analysis(self,udict):
        uvec = np.zeros((2*(self.n-1)), dtype=complex)
        ct = 0
        for ukey in udict:
            uvec[ct] = float(udict[ukey])
            ct += 1
        uin2 = np.reshape(uvec,(self.n-1,2))
        cost = self.value_function(uin2)
        if sum(uvec).imag == 0.0:
            print(f"cost = {cost}")
        funcs = {"obj" : cost}
        return funcs

    def gradient(self,udict,funcs):
        grad = np.zeros((2*(self.n-1)))
        eps = 1e-30
        uinput = np.zeros((2*(self.n-1)), dtype=complex)
        ct = 0
        for ukey in udict:
            uinput[ct] = float(udict[ukey])
            ct += 1
        utemp = np.zeros((2*(self.n-1)), dtype=complex)
        for i in range(2*(self.n-1)):
            if i < 2*(self.n_fixed-1):
                grad[i] = 0.0
            else:
                utemp[:] = uinput[:]
                utemp[i] += 1j * eps
                uin2 = np.reshape(utemp,(self.n-1,2))
                grad[i] = self.value_function(uin2).imag/eps
        sens = {}
        sens["obj"] = {}
        ct = 0
        for ukey in udict:
            sens["obj"][ukey] = grad[ct]
            ct += 1
        #print(f"utemp = {utemp}")
        return sens