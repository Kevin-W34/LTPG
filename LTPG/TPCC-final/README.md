"Random.h" is used to generate random numbers.

"Database.h" is used to generate the database. We set up copying the databasenfrom the CPU to GPU in this file.

"Genericfunction.h" is used to set up general function, such as timekeeping function, execution function, checking function and write-back function.

"Execute.h" is used to implement specific functions in transactions.

"Datastructure.h" is used to define the data structure used in LTPG system.

"Query.h" is used to randomly generate new transactions in LTPG system.

"Predefine.h" is used to define system execution parameters.

"tp.cu" is used to implement the main function to run the system.

# Build

bash compile.sh

# Execution

./tp