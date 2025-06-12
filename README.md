# EVA-S3PC: Secure-3-party-Computation-Framework
A decentralized framework for secure multi-party matrix computation, including various secure 2-party and 3-party operators, which is well-suited for scientific computing and machine learning.

## 1. Framework
A S3PC call from a client requests the computation of a function $f(A,B,C)$ on data $A$, $B$, $C$ supplied by isolated data owners without disclosing any information about individual data to anyone else except for the owner. In this paper, we assume that the operands are organized in the form of matrices. Computation proceeds by invoking the basic protocols in the Secure Protocol Library (SPL). Each basic protocol consists of Commodity-Server (CS) pre-processing, Online Computing and Result Verification.

### Pre-processing Stage
CS, initially introduced by Beaver as a semi-trusted third party, is widely utilized to address various S2PC problems. It simply generates offline random matrices $(R_a,r_a),(R_b,r_b)$ and $(R_c,r_c)$ to be used by the subsequent online computing stage to disguise original data, and is prohibited from conspiring with any other participants. Its simplicity feature makes it easy to be built in real world applications.

### Online Computing Stage
The online computing stage following CS pre-processing computes $f(A,B,C)$ by sequentially executing the fundamental protocols (S2PM, S3PM, S2PI, S2PHM, S3PHM) among data owners who can only access partial outputs $(\beta_a,\beta_b,\beta_c)$ with which no information about the input and output data can be inferred. 

### Result Verification Stage
Because the request client normally does not participate in the calculation of $f(A,B,C)$, there ought to be a mechanism to check the reliability of the output to ensure that its computation follows the exact protocol. In the result verification stage, participants collaborate to produces check matrices ($VF_a$, $VF_b$, $VF_c$) respectively and can independently use them to check the reliability of the output result.

## 2. Project Directory Structure

Below is the directory structure of the EVA-S3PC:

```plaintext
S3PMC_Framework/
├── Basic_Subprotocol/
│   ├── Efficient_Evaluation/
│   ├── Precision_Evaluation/
│   └── Protocols/
└── S3PLR/
    ├── data/
    ├── accuracy_analysis.py
    ├── S3PLR.py
    └── time_cost.py
```

### Directory Description

#### Basic_Subprotocol

The `Basic_Subprotocol` directory provides the implementation and evaluation of basic protocols (S2PM, S3PM, S2PHM, S3PHM, S2PI):

- **`Basic_Subprotocol/Protocols`**: Contains the source code for the basic protocols.
- **`Basic_Subprotocol/Efficient_Evaluation`**: Contains the efficiency evaluation of the basic protocols.
- **`Basic_Subprotocol/Precision_Evaluation`**: Contains the precision evaluation of the basic protocols.

---

#### S3PLR

The `S3PLR` directory provides the implementation code and experimental evaluation of secure three-party linear regression:

- **`S3PLR/S3PLR.py`**: Contains the source code for the secure three-party linear regression model.
- **`S3PLR/data`**: Contains the dataset used for experimental evaluation.
- **`S3PLR/accuracy_analysis.py`**: Provides the accuracy analysis of various metrics for the model.
- **`S3PLR/time_cost.py`**: Provides the time cost evaluation of the model.

## 3. Performance Evaluation

### Choice of Parameters
#### Basic Protocols
Input matrices for running time testing in 2-party and 3-party protocols are Float 64 (ring size $l=64$), with elements sampled from $x \in [10^{-4}, 10^{4}]$ and represented by a 15-bit significant figure "$1.a_1a_2\cdots a_{15}\times 10^\delta (\delta\in \mathbb{Z}=[-4,4]])$". Precision tests are divided into 6 ranges, from $[E0, E0]$ to $[E-10, E+10]$, with exponents incremented by $\Delta=2$.

#### S3PLR
- **`Datasets`**: The **Boston dataset** consists of 506 samples (404 for training, 102 for testing), with 13 features and 1 label. The **Diabetes dataset** consists of 404 samples (353 for training, 89 for testing), with 10 features and 1 label. In S3PLR, the label is private and only accessible to Carol, while the features are evenly split between Alice and Bob.

- **`Metrics`**: Accuracy comparison metrics include mean absolute error (MAE), mean square error (MSE) and root mean square error (RMSE) between prediction and label, L2-Norm relative error (LNRE) between securely trained model parameters and the ones learned from the plain text data using Scikit-learn, R-Square, and R-Square relative error (RRS) between privacy preserving models and plain text models. Particularly, LNRE measures the relative error between securely trained parameters $\beta$ and the ones learned from plaintext using Scikit-learn $\widehat{\beta}$: $LNRE = \frac{||\beta - \widehat{\beta}||_2}{||\beta||_2}$. RRS quantifies the relative difference of the R-Square values between secure and plaintext models: $RRS = \frac{|R^2 - \widehat{R}^2|}{\widehat{R}^2} \times 100\%$.
 
### Communication
The communication generated during the execution of the protocol (the communication part can be implemented using a framework of choice) can be captured using the Wireshark tool, primarily to record the communication volume and the number of communications. Theoretically, the communication time follows the formula:

$$
Time = Volume / Bandwidth + Count \times Latency
$$

## 4. Installation
EVA-S3PC is implemented in Python and primarily depends on the `numpy` package, which can be installed via pip:

```bash
pip install numpy
```

