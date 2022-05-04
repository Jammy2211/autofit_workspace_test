import numpy as np
import pickle

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


### FIXED REDSHIFT ##

name = 'dynesty_review/widesubpos/mask3013_NARROWWIDESUBPOS_on_jvas_70pixSUBTR_n_max10SUBPJAFFE'

obj = load_obj(name)
print(f"With Fixed Redshift PJAFFE= {np.max(obj.logz)}")



### REDSHIFT FREE ###

name = "dynesty_review/narrowprior/mask3013_NARROW_on_jvas_70pixSUBTR_n_max10INTNFW"

obj = load_obj(name=name)
print(f"With Interpolator NFW = {np.max(obj.logz)}")


name = "dynesty_review/wideprior/mask3013_on_jvas_70pixSUBTR_n_max10INTPJAFFE"

obj = load_obj(name=name)
print(f"With Interpolator PJaffe = {np.max(obj.logz)}")

name = "dynesty_review/wideprior/mask3013_on_jvas_70pixSUBTR_n_max10INTSIS"

obj = load_obj(name=name)
print(f"With Interpolator SIS = {np.max(obj.logz)}")

### CONCENTRATION FREE ###

name = "dynesty_review/freeconcen/mask3013_NARROWCONCEN_on_jvas_70pixSUBTR_n_max10SUBcNFW"

obj = load_obj(name=name)
print(f"With Free C NFW = {np.max(obj.logz)}")

name = "dynesty_review/freeconcen/mask3013_NARROWCONCEN_on_jvas_70pixSUBTR_n_max10INTcNFW"

obj = load_obj(name=name)
print(f"With Interpolator and Free C NFW = {np.max(obj.logz)}")

name = "dynesty_review/freeconcen/mask3013_NARROWCONCEN_on_jvas_70pixSUBTR_n_max11INTcNFW"

obj = load_obj(name=name)
print(f"With Interpolator and Free C NFW (11 shapelets) = {np.max(obj.logz)}")