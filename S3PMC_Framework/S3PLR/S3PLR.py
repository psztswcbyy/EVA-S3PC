import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

from Basic_Subprotocol.Protocols.S2PM import S2PM
from Basic_Subprotocol.Protocols.S2PHM import S2PHM
from Basic_Subprotocol.Protocols.S2PI import S2PI
from Basic_Subprotocol.Protocols.S3PHM import S3PHM


# Secure 3-Party Linear Regression Model
class S3PLR():
    # Training Phase
    def fit(self, X1, X2, Y):
        X1_T = X1.T
        X2_T = X2.T
        V = S2PHM([X1_T, X1], [X2_T, X2], 1)
        U = S2PI(V[0], V[1], 1)
        res = S3PHM([U[0], X1_T], [U[1], X2_T], Y, 1)
        self.Wa, self.Wb, self.Wc = res[0], res[1], res[2]

    # Predicting Phase
    def predict(self, X1, X2):
        V1 = S2PHM([X1, self.Wa], [X2, self.Wb], 1)
        V2 = S2PM(X1, self.Wc, 1)
        V3 = S2PM(X2, self.Wc, 1)
        Ya = V1[0] + V2[0]
        Yb = V1[1] + V3[0]
        Yc = V2[1] + V3[1]
        return Ya, Yb, Yc