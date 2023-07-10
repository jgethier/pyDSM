import math

def pdflognormal(x,Mn,Mw):
    prob=math.exp(-(math.log(x)-math.log(Mn**(3/2)/math.sqrt(Mw)))**2/(4*math.log(math.sqrt(Mw/Mn))))
    norm=2*x*math.sqrt(math.pi*math.log(math.sqrt(Mw/Mn)))
    return prob/norm

def froot(Mmax, Mn, Mw):
    fmax=pdflognormal(Mmax,Mn,Mw)
    Mmp=Mn**(5/2)/Mw**(3/2)
    fmp=pdflognormal(Mmp,Mn,Mw)
    return fmax - 10**(-4)*fmp