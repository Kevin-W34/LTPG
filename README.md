# LTPG
Large-batch transaction processing on GPUs with deterministic concurrency control

Regarding the GaccO code, we have contacted the authors and obtained the source code from them. 
The experimental evaluation in our paper is conducted using the obtained codes. 
However, they asked not to publicly disclose their source code, so we have retained our reproduced version for reference in the repository. 
As for the GPUTx code, since the source code is not available, we have provided our reproduced version.

# Discription of files in the project

"Random.h" is used to generate random numbers.

"Database.h" is used to generate the database. We set up copying the databasenfrom the CPU to GPU in this file.

"Genericfunction.h" is used to set up general function, such as timekeeping function, execution function, checking conflicts function and write-back function.

"Execute_neworder.h" is used to implement specific functions in neworder transactions.

"Execute_payment.h" is used to implement specific functions in payment transactions.

"Datastructure.h" is used to define the data structure used in LTPG system.

"Query.h" is used to randomly generate new transactions in LTPG system.

"Predefine.h" is used to define system execution parameters.

"tp.cu" is used to implement the main function to run the system.

# Build

bash compile.sh

# Execution

./tp
