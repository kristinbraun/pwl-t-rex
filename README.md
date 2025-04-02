# PWL-T-Rex
## Piecewise Linear Tool for Relaxations

```text                                                                        
                       @@   @@                                            
                    @           @                                         
                  @@@@@          @                                        
                @          @@     @                                       
              @      @@        @@ @                                       
            @             @@    @@@@                                      
        @@@       @@@@        @@@   @                                     
     @            @  @@         @   @                                     
   @              @@@@           @                                        
  @   @@                         @                                        
  @                              @                                        
  @                 @                                                     
   @     @@  @   @          @    @                                        
    @ @@                           @@                                     
     @                    @                @@@                            
       @     =@@@@@@@@@                         @@                        
                      @ @@                         @                      
                       @ @@             PWL         @                     
                        @               -T-           +                   
                       -@ @             Rex           @                   
                   @@@     @     @                     @                  
                  @@@  @@@@      @      *               @          @+     
                    @  @@      @@      @                  @        @@     
                      @@  @@@     @    @            @       @@@ :@@       
                     @:@@:  @@       @@             @               @     
                               @                   @               @@     
                              @  @     @           @@@           @@@      
                                 @  @@  @           @      @@@   @        
                                   @      @         @@@@@  +@@@           
                                            @@      @                     
                                @        @           @                    
                                        @    @       @                    
                              @@@@@@@@@    @        @                     
                                    @     @@       @                      
                                         @@@@@@@@@*                       
                                               -                          
```

This project implements and compares different Mixed Integer Programming (MIP) representation methods for reformulating Mixed-Integer Nonlinear Programs.

## Features

- Multiple MIP formulation methods:
  - Disaggregated Convex Combination
  - Logarithmic Disaggregated Convex Combination
  - Aggregated Convex Combination
  - Logarithmic Aggregated Convex Combination
  - Incremental Method
  - Multiple Choice Model
  - Binary Zig-Zag Model
  - Integer Zig-Zag Model

## Installation

1. Clone this repository:
```bash
git clone https://github.com/kristinbraun/pwl-t-rex.git
```

2. Install the required dependencies using the provided requirements.txt:
```bash
pip install -r requirements.txt
```

### Dependencies

The following packages are specified in requirements.txt:
- pyomo>=6.0 - Mathematical optimization modeling
- gurobipy>=10.0 - Gurobi optimization solver interface
- numpy>=1.20 - Numerical computing
- scipy>=1.7 - Scientific computing utilities
- beautifulsoup4>=4.9 - XML/HTML parsing
- lxml>=4.9 - XML/HTML processing library

External software requirements (optional for file conversion):
- SCIP - Mixed Integer Programming solver
- GAMS - Modeling system for mathematical optimization
  - Academic licenses available at www.gams.com

Note: These external tools are only needed if you need to convert input files. If you're working directly with .osil files, you can skip installing them.

## Usage

Run the main script with your desired parameters:

```bash
python pwltrex.py [filename] [options]
```

### Command Line Arguments

- `filename`: Path to the input file (required)
- `--method`: MIP method selection (-5 to 8)
  - -5: Run all MIP reformulations
  - -1: Initial MINLP
  - 0: 1D-MINLP
  - 1: DisaggConvex
  - 2: LogDisaggConvex
  - 3: AggConvex
  - 4: LogAggConvex
  - 5: Incremental
  - 6: MultipleChoice
  - 7: BinaryZigZag
  - 8: IntegerZigZag
- `--epsilon`: Error tolerance (default: 1)
- `--relax`: Relaxation method (0: Approximation, 1: Exact error, 2: Fixed error)
- `--timelimit`: Time limit for MILPs in seconds (default: 60)
- `--create`: Create model without solving (0: No, 1: Yes)
- `--solver_output`: Print solver output (0: No, 1: Yes)


### Example

```bash
python pwltrex.py alkyl.osil --method -5 --epsilon 0.1 --timelimit 120
```


### File Conversion (Optional)

If your input problems are in AMPL/GAMS format rather than .osil format, you can use the conversion script in the `instances/convert` folder. This entire conversion step is optional and only needed if you're not working directly with .osil files.

```bash
# Make the conversion script executable
chmod +x instances/convert/convert_nl_to_osil.sh

# Convert from AMPL/GAMS format
./instances/convert/convert_nl_to_osil.sh input

# The converted .osil file can then be used as input:
python pwltrex.py input.osil --method -5
```

Note: The script will automatically handle the file extension, so you only need to provide the base filename. If you already have .osil files, you can skip this entire conversion section and its dependencies.

Errors regarding newlines, i.e. LF/CRLF, (e.g. `'\r': command not found`) can be handled by using one of the two commands:
``` bash
# Converting from Windows to Linux file format
dos2unix instances/convert/convert_nl_to_osil.sh

# Converting from Linux to Windows file format
unix2dos instances/convert/convert_nl_to_osil.sh
```


## Output

The program outputs:
- Objective function value
- Runtime to optimal and to first primal solution
- Sorted rankings of methods by:
  - Total runtime
  - Time to first primal solution


## Project Structure
- `pwltrex.py`: Main execution script
- `MIPRef_osilToOnedim.py`: Converts OSIL format to one-dimensional representation 
- `MIPRef_onedimToMIP.py`: Converts one-dimensional to MIP representation
- `MIPRef_mipRepresentations.py`: Contains MIP formulation methods
- `MIPRef_graycode.py`: Gray code utilities for binary encoding
- `MIPRef_linrelax.py`: Linear relaxation utilities
- `nltree.py`: Nonlinear expression tree implementation
- `evaluation_solving.py`: Solving utilities
- `evaluation_statistics.py`: Statistical analysis utilities
- `instances/convert/`: Utilities for converting AMPL/GAMS files to OSIL format
  - `convert_nl_to_osil.sh`: Script for converting .nl files to .osil format
  - `add_optline.py`: Python helper file for conversion


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

## Citation

If you use PWL-T-Rex in your research, please cite:

Braun, K., & Burlacu, R. (2023). A Computational Study for Piecewise Linear Relaxations of Mixed-Integer Nonlinear Programs. *Optimization Online*. Retrieved from https://optimization-online.org/2023/09/a-computational-study-for-piecewise-linear-relaxations-of-mixed-integer-nonlinear-programs/

BibTeX:
```bibtex
@article{braun2023computational,
    title={A Computational Study for Piecewise Linear Relaxations of Mixed-Integer Nonlinear Programs},
    author={Braun, Kristin and Burlacu, Robert},
    journal={Optimization Online},
    year={2023},
    url={https://optimization-online.org/2023/09/a-computational-study-for-piecewise-linear-relaxations-of-mixed-integer-nonlinear-programs/}
}
```

