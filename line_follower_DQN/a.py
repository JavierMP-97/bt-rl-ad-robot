import numpy as np
import pickle



    with open( "calibration.pr" , "wb" ) as f:
        pickle.dump( infra_red_range, f, protocol=pickle.HIGHEST_PROTOCOL )