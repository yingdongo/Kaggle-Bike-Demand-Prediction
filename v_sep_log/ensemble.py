# -*- coding: utf-8 -*-
"""
Created on Thu May 28 16:40:00 2015

@author: Ying
"""
import pandas as pd
import numpy as np
pred1=pd.read_csv("submissions/paramtuning_solution_gbr700.35x2_month.csv")#0.36899
pred2=pd.read_csv("submissions/paramtuning_solution_ex90010_ex100010_month.csv")#0.38196
pred3=pd.read_csv("submissions/paramtuning_solution_rf6008_rf5008_month.csv")#0.38195
pred4=pd.read_csv("submissions/paramtuning_solution_gbr700.35_ex100010_month.csv")#0.36959
pred5=pd.read_csv("to_submit/paramtuning_solution_gbr700.35_rf5008_month.csv")#0.36832
pred6=pd.read_csv("submissions/paramtuning_solution_ex90010_gbr800.33_month.csv")#0.37745
pred7=pd.read_csv("submissions/paramtuning_solution_ex90010_rf5006_month.csv")#0.38005
pred8=pd.read_csv("to_submit/paramtuning_solution_rf6008_gbr800.38_month.csv")
pred9=pd.read_csv("to_submit/paramtuning_solution_rf6008_ex90010_month.csv")

#pred=np.around((pred1["count"]+pred2["count"]+pred3["count"]+pred4["count"]+pred5["count"]+pred6["count"]+pred7["count"]+pred8["count"]+pred9["count"])/9.0)
#pred=np.around((pred1["count"]+pred4["count"]+pred5["count"]+pred6["count"])/4.0)

#pred=np.around((pred1["count"]*2.33866329e-01+pred2["count"]*-7.09082595e-02+pred3["count"]*2.24565375e-17+
#    pred4["count"]*1.79635279e-01+pred5["count"]*2.04723916e-01+pred6["count"]*-1.25767452e-17
#    +pred7["count"]*1.10315011e-01+pred8["count"]*2.00551206e-01+pred9["count"]*5.25838054e-18))
#pred=np.around((pred1["count"]*0.22028047+pred4["count"]*0.23386754+pred5["count"]*0.2437300+pred6["count"]*0.30212191))

pred=np.around((pred1["count"]+pred4["count"]+pred5["count"])/3.0)

submission=pd.concat([pred1["datetime"], pred], axis=1)

submission.to_csv("to_submit/ensemble_3.csv", index=False)

#Best Weights: [  2.33866329e-01   7.09082595e-02   2.24565375e-17   1.79635279e-01
#   2.04723916e-01  -1.25767452e-17   1.10315011e-01   2.00551206e-01
#   5.25838054e-18]

# 0.22028047  0.23386754  0.24373008  0.30212191
#0.36830	3 average