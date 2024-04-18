# Create the 1 billion row file as measurements.txt (~14GB)
python data/create_measurements.py 1000000000

# Compile with HIP and run
hipcc -o fast_hip fast_hip.cpp

time ./fast_hip data/measurements.txt 1000000 600000 
